# vggtSam 对话接力上下文

本文件用于从旧 Codex 长对话
`019ef324-d8d7-78f3-a234-3df280b150c4` 接续工作。新对话应先读取本文件、
`streaming_couping/docs/method.md`、`streaming_couping/commands.txt` 和当前
Git 工作区，不要从头重新设计或重复已经完成的实验。

## 项目目标

研究 SAM3 与 StreamVGGT 的双向协同：

- 几何帮助跟踪：解决大视角变化、遮挡后重现和相似外观下的实例恢复。
- 稳定实例帮助几何：把长期可靠实例作为 object-level factors，约束相机漂移。

当前优先完成第一条，并建立可解释、可消融的 baseline。

## 已形成的核心判断

合理的职责分工是：

```text
SAM3：识别语义类别并产生完整物体边界
StreamVGGT：把历史实例重定位到当前帧的大致区域
原 SAM3 obj_id：维持实例身份并承接 memory
```

当前 baseline 流程：

```text
SAM3 正常跟踪
  -> 实例缺失或面积塌缩
  -> 历史 3D object map 投影到当前帧
  -> SAM3 在全图生成同语义类别的完整候选 masks
  -> 几何支持区域从同类候选中选择历史实例
  -> 完整 mask 写回原 obj_id
  -> SAM3 memory 继续传播
  -> 可靠新视角再扩充该实例的 3D object map
```

几何不应使用局部 point/box 限制完整边界。对于首帧只看到床的一部分等情况，
SAM3 应负责生成完整语义候选，几何只负责实例对应。

## 已有实验结论

固定 ScanNet++ 场景 `00a231a370`、实例 37 和帧
`[133, 162, 520, 566, 477]` 的 token 诊断：

- SAM3 FPN2：Top-1 0.50，margin 0.129，定位误差 0.108。
- SAM3 actual spatial memory：Top-1 0.50，定位误差 0.287。
- SAM3 oracle spatial memory：Top-1 1.00，定位误差 0.051。
- StreamVGGT layer 17：Top-1 1.00，margin 0.295，定位误差 0.032。
- StreamVGGT layer 23 并非最佳层；layer 4/11 的实例定位能力较弱。

这已经证明 Layer 17 含稳定跨视角对应信息。不要再把“证明 Layer 17 有匹配信息”
本身当作主要贡献。此前直接做 SAM3/VGGT token 融合失败，可能来自语义模态、
坐标预处理和最终 gate/memory 决策三重不对齐。

## 已实现代码

- `streaming_couping/src/recovery_selection_ablation.py`
- `streaming_couping/src/bridge/segment_descriptor.py`
- `streaming_couping/scripts/run_recovery_selection_ablation.py`

已实现候选排序模式：

- `geometry_only`
- `descriptor_only`
- `geometry_descriptor`
- `shuffled_descriptor`

四个分支复用同一 SAM3 候选、StreamVGGT 输出、恢复门控和 obj_id 写回；Layer 17
默认保持原生 `[T,2048,22,37]` 网格；GT 只用于诊断。代码已通过静态语法检查，
但本机没有 PyTorch/GPU，真实实验需在远端 `3am` 环境运行。

这套代码可保留为候选选择诊断，但用户指出：若只复现“Layer 17 能匹配”，会重复
旧结果，意义不足。继续实验前应先判断失败发生在候选生成、候选选择，还是 memory
写回后的传播。

## 下一步实验原则

首先建立以下最小因果消融，恢复帧使用完全相同的完整 mask，唯一变量是是否写回：

1. `SAM3 original`
2. `geometry recovery`：只修复当前帧，不写 memory
3. `geometry recovery + same-ID memory writeback`
4. `shuffled geometry + same-ID memory writeback`

评估：

- 恢复帧 IoU
- 恢复后可见帧 IoU
- 漏检率
- 不可见帧误检率
- obj_id 是否保持

同时输出候选 oracle 诊断：

- SAM3 候选数量
- 最佳候选 GT IoU（oracle 上限）
- 几何选中候选的 GT IoU
- 几何是否选中 oracle
- 正确候选写回后是否继续保持实例

只有“oracle 候选良好但几何选错”时，才需要继续 Layer-17 descriptor 排序；若候选
本身没有完整物体，应优先改全图语义候选生成；若候选正确但后续又丢失，应修复
presence gate、memory 写回或传播。

## 当前用户待办

旧线程最后连续询问：“现在我要运行什么命令来消融实验呢？”

新线程接续后应：

1. 先检查当前实现与 `commands.txt`，确认上述四组最小因果消融是否已经完整接线。
2. 若尚未接线，直接补齐代码、CSV 字段和统一运行入口。
3. 给出一条可在远端 `3am` 环境直接运行的命令。
4. 不要重新做已经失败的直接 token 融合，也不要仅重复 Layer-17 特征诊断。

测试场景优先使用实例 54（bed）的压力序列：

```text
scene: 00a231a370
frames: 90 105 119 130 140 210 240
SAM3 device: cuda:3
geometry device: cuda:1
manifest: data/processed/scannetpp_pinhole_2d/manifest.json
```

旧会话原始记录仍保存在：

```text
~/.codex/sessions/2026/06/23/
rollout-2026-06-23T14-22-18-019ef324-d8d7-78f3-a234-3df280b150c4.jsonl
```

## 2026-07-18 更新：一次性完整消融

直接特征融合与单实例 ICP 路线继续暂停。当前服务器实验已扩展成一次性双向机制
消融，正式设计见 `docs/ablation_plan.md`，唯一命令见 `commands.txt`。

新增内容：

- `natural_joint_gate` 与固定干预 `scheduled_probe` 同次运行；
- non-empty/high-score mask 与 3D 支持冲突时也可触发恢复；
- `reference_only` 对照与 `joint_reliable` 历史 mask 扩图；
- aligned、shuffled、oracle candidate、oracle mask 七分支；
- 所有后续帧 global-text candidate 缓存与候选上限诊断；
- 27 组 gate 阈值的 post-hoc sweep；
- reference 固定 Sim(3) 下的对象地图 5/10 cm precision、recall、F-score
  与 Chamfer；GT 几何仅用于评估；
- 实例点云 PLY 继续保留。

当前运行实例为 `37 68 54`。37/68 是不应误伤的易例，54 bed 是检验
“高置信跟错”与 geometry-disagreement gate 的压力例。运行结束后优先回传：

```text
summary.csv
candidate_screening.csv
threshold_sweep.csv
geometry_gate_diagnostics.csv
map_quality.csv
metadata.json
完整日志
```
