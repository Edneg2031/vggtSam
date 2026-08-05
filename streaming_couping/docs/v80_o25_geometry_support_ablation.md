# V8 O2.5：跨帧支持率与预测几何上界

O1/O2 已确认 30/30 帧坐标审计通过，但暴露了两个问题：medium 每帧只有 2–4 个 GT
伪对应；StreamVGGT depth 反投影与 point head 的平均 P90 差异约 0.55 m。O2.5 不训练任何
网络，也不改变 SAM3.1/StreamVGGT cache，只分别诊断“对应支持不足”和“预测几何不一致”。

## 一键运行

```bash
zsh streaming_couping/commands_v80_geometry_support_ablation.txt
```

它只读取已有 V7.4 长序列 cache，在 CPU 上运行小规模 Kabsch，不加载两个 backbone。

## Support sweep

只使用 O1 GT camera geometry，组合：

- local token：32 / 64；
- 每个 persistent slot 的严格因果历史：最近 1 / 2 / 4 次有效写入；
- GT-world 伪对应：mutual NN / one-way NN；
- 半径：0.10 / 0.15 m；
- weighted / 70% trimmed Kabsch。

当前帧读取历史 bank 后才写入自身，所有历史索引严格小于当前帧。多历史允许同一个当前点在
不同历史帧形成约束，但 `effective_correspondences` 会反映权重集中或重复造成的有效样本减少。

## 固定支持下的几何来源

主表预先固定 `K=64、history=4、one-way、radius=0.15m`，不会根据三个 test fold 选择配置：

| 分支 | 当前/历史 camera point | 历史位姿 | 定位 |
|---|---|---|---|
| O1 | GT geometry | GT | solver/support 上界 |
| O2-GT | StreamVGGT depth | GT | depth 几何上界 |
| O2-L0 | StreamVGGT depth | L0 | 实际历史漂移 |
| O2P-GT | point head 经 L0 W2C 转 camera | GT | pose-entangled 诊断 |
| O2P-L0 | point head 经 L0 W2C 转 camera | L0 | L0 自洽/循环诊断 |

O2P 不是独立位姿证据，因为它使用 L0 W2C 把共享 pointmap 转回 camera frame，不能作为
“point head 改善 L0”的正式结论。

## 三层门控

每个配置同时报告：

1. `bounded`：原 correspondence、退化、RMSE、extent、rotation/center correction 门控；
2. `reliable`：再要求有效对应不少于 6、唯一当前点不少于 6，且 depth/point-head P90
   一致性不超过 0.60 m；多历史重复使用同一当前点不会虚增空间支持；
   O1 GT geometry 不受 head-consistency 限制；
3. `consensus`：再要求至少两个实例能独立求解，candidate 之间旋转不超过 5°、center 不超过
   0.25 native。只有一个实例的帧会安全回退。

所有门控输入都来自预测输出、对应统计和拟合结果，不读取 GT pose error。GT 只用于实验后评分。

`*_robust_pass=1` 必须同时满足：

- 每个四帧 fold 至少 3 帧生效；
- 汇总 pose loss 至少提升 1%；
- 没有任何生效帧比 L0 更差；
- 所有未生效帧 bit-exact 回退 L0。

## 输出

```text
outputs/streaming_couping_v80_geometry_support_ablation/
├── v80_o25_geometry_ablation.csv       # 30 行紧凑主表，优先复制
├── v80_o25_support_ablation.csv        # 完整 O1 support sweep
├── v80_o25_frame_diagnostics.csv       # 每帧 gate/reason/candidate
├── v80_o25_decision.md
└── v80_o25_metadata.json
```

判定顺序：先看 support sweep 是否存在 short/medium/long 全通过配置；再看固定支持下 depth O2
是否全 fold 通过。只有 depth O2 形成稳定显式位姿上界后，才进入 SAM descriptor correspondence
的 O3，避免再次让 learned pose head 掩盖几何问题。
