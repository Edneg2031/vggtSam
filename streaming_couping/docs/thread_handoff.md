# vggtSam 对话接力上下文

本文件用于从旧 Codex 长对话
`019ef324-d8d7-78f3-a234-3df280b150c4` 接续工作。新对话应先读取本文件、
`streaming_couping/docs/method.md`、`streaming_couping/commands.txt` 和当前
Git 工作区，不要从头重新设计或重复已经完成的实验。

## 当前状态（2026-07-20，最新）

几何辅助 SAM3 实例恢复阶段已经完成。实例 54（bed）在高置信错误追踪压力序列
上由 `natural_joint_gate` 自动于帧 119 恢复；37（cabinet）和 68（wardrobe）
两个易例没有被误触发。当前不继续做 held-out 扩展，也不继续调整这条序列的
阈值。已验证配置保存在：

```text
streaming_couping/configs/recovery_050_025.yaml
```

服务器原始结果目录为：

```text
outputs/streaming_couping_threshold_050_025_probe119_37_68_54/
```

固定阈值为：

```yaml
tracker_min_geometry_coverage: 0.50
recovery_min_support_coverage: 0.25
map_update_min_geometry_coverage: 0.50
```

SAM3 恢复路线现标记为“阶段完成/暂停”。下一阶段转向：

1. 实例 mask 条件下的 StreamVGGT 点云生成、融合与质量分析；
2. StreamVGGT 相机位姿的坐标约定、ATE/RPE 与漂移诊断；
3. 研究可靠静态实例点云能否为相机位姿提供约束。

第一步 raw baseline 已实现于 `src/pose_pointmap_diagnostics.py`，当前服务器命令
在 `commands.txt`。该诊断不运行 SAM3，不使用实例 mask，只输出 frozen
StreamVGGT 的 pose、pointmap 与 intrinsics evaluation。

不要因为后续研究点云/位姿而删除当前 same-ID memory writeback 流程；它是已经
验证过的可靠实例 mask 来源。

## 项目目标

研究 SAM3 与 StreamVGGT 的双向协同：

- 几何帮助跟踪：解决大视角变化、遮挡后重现和相似外观下的实例恢复。
- 稳定实例帮助几何：把长期可靠实例作为 object-level factors，约束相机漂移。

第一条已经完成当前压力测试下的阶段性验证；现在优先进入第二条的点云与相机位姿
研究。

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

## 历史用户待办（已完成）

旧线程最后连续询问：“现在我要运行什么命令来消融实验呢？”

以下事项已在 2026-07-20 前完成，仅作为历史记录：

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

## 2026-07-20 更新：恢复阶段最终结果

### 最终方法

```text
SAM3 原始实例追踪
  -> 与时序对齐的 StreamVGGT object-map 投影计算 geometry coverage
  -> coverage < 0.50 时触发全图文本候选
  -> 按 3D support coverage 选择同类实例
  -> selected support coverage >= 0.25 时接受
  -> 完整候选 mask 写回原 SAM3 obj_id memory
  -> 后续可靠 mask 继续生成实例点云
```

SAM3 和 StreamVGGT 均冻结。reference 后的 GT 不参与 natural trigger、候选排序、
memory writeback 或 object-map 更新。

### 2D 跟踪结论

bed 在帧 119 的原始 SAM3 score 为 `0.9844`，但 IoU 只有 `0.0207`，证明仅用
score 不能识别高置信跟错。几何分支选中的完整文本候选 IoU 为 `0.9323`，且与
候选 oracle 完全一致。

| bed 分支 | cross-view IoU | 帧 119 IoU | 后续 4 帧平均 IoU |
|---|---:|---:|---:|
| original | 0.1292 | 0.0207 | 0.0002 |
| geometry recovery，无 memory | 0.2812 | 0.9323 | 0.0002 |
| geometry recovery + same-ID memory | 0.7986 | 0.9323 | 0.7763 |
| shuffled geometry + memory | 0.1292 | 0.0207 | 0.0002 |
| GT-visible-mask writeback control | 0.6370 | 1.0000 | 0.5170 |

same-ID memory 相比 no-memory 的 post-recovery IoU 增益为 `+0.7762`。恢复后的
逐帧 IoU 为：帧 130 `0.8674`、帧 140 `0.9487`、帧 210 `0.6787`、帧 240
`0.6105`。natural gate 与 scheduled probe 得到完全相同的 bed 结果。

37/68 的 natural gate 均未触发，所有分支逐帧 IoU 与 original 完全相同。
shuffled geometry 的候选支持覆盖不足而被拒绝，因此提升不能解释为“任意调用一次
SAM3 文本分割都会改善”。

### 实例点云结论

使用一次固定 reference-frame Sim(3) 进行 evaluation-only 对齐：

| bed map 分支 | Chamfer-L1 ↓ | F-score@5cm ↑ | F-score@10cm ↑ |
|---|---:|---:|---:|
| original | 0.5269 | 0.0359 | 0.0819 |
| recovery，无 memory | 0.2678 | 0.0836 | 0.2167 |
| recovery + same-ID memory | 0.0416 | 0.7152 | 0.9443 |
| shuffled geometry | 0.5269 | 0.0359 | 0.0819 |
| GT-mask map oracle | 0.0403 | 0.7396 | 0.9481 |

完整方法的 F-score@10cm 达到 `0.9443`，接近 GT-mask map oracle 的 `0.9481`；
Chamfer 从 `0.5269` 降到 `0.0416`。这证明恢复后的可靠追踪能显著改善实例点云。

`reference_geometry_same_id_memory` 与 `geometry_recovery_same_id_memory` 在当前
序列中结果相同。因此已经证明的是：

- 时序对齐几何帮助追踪恢复；
- 恢复后的追踪帮助实例点云构建。

尚未证明“可靠历史 tracking 扩充的 3D map 比 reference-only map 更能改善候选
选择”。该边界必须保留，不能把单例结果写成泛化结论。

恢复帧直接写入 GT 可见 mask 的未来传播和地图结果反而低于完整文本候选，因此
代码中的 `oracle_mask_same_id_memory` 仅保留为兼容名称；论文表述应使用
“GT-visible-mask writeback control”，不能称为未来传播上限。

### 阶段决策

本结果被记录为当前压力测试下的阶段性成功，不继续 held-out 扩展。后续不再恢复
token fusion、descriptor 或旧 translation-only ICP 路线，除非新的点云/位姿
诊断给出明确需求。下一研究入口为：

- `src/instance_point_cloud.py`：实例点云导出；
- `src/instance_map_evaluation.py`：固定 Sim(3) 下的 3D 指标；
- `src/backbones/streamvggt_wrapper.py`：pointmap、world-to-camera、intrinsics；
- `src/aggregation/point_map_fusion.py`：共享坐标系中的点图融合。
- `src/pose_pointmap_diagnostics.py`：raw pose ATE/RPE、pointmap 和内参诊断。

## 2026-07-20 更新：pointmap-consistent camera translation

raw 诊断已经完成。固定 reference pose、沿用 reference point scale 后：

- ATE RMSE `0.2280 m`；
- all-pairs rotation mean `2.39°`，rotation@5° `100%`；
- translation-direction mean `14.56°`，@5° `19.0%`；
- `105->119` 的 GT/预测运动为 `0.260/0.300 m`，方向误差 `34.33°`；
- `210->240` 的 GT/预测运动为 `0.123/0.312 m`，方向误差 `44.22°`；
- 帧 240 pointmap RMSE `0.1866 m`，camera-center error `0.460 m`；
- predicted `fx/fy` 全部偏大，但焦距误差与 pointmap RMSE 不同步。

结论是 rotation 较稳，translation 同时存在局部方向和尺度错误；单一全局尺度、
时间平滑或旧单实例 ICP 都不是充分修复。当前实现先隔离验证 point head 是否能
修正 camera head：

```text
StreamVGGT pointmap X_i + pixel ray d_i(predicted K/R)
  -> min_C sum_i w_i ||(I-d_i d_i^T)(X_i-C)||^2
  -> confidence gate 后使用全部射线
  -> keep R, replace t with -R C
```

服务器结果选定可部署主分支 `ray_predicted_k_all`，不使用 GT；80% trimmed
分支在所有非 reference 帧均略差，只保留为消融。同一次还输出 GT-K、GT-R、
GT-K+R 和 spatially-shuffled pointmap 分支。GT 分支只诊断 intrinsics/rotation
上限，shuffled 是破坏像素—点对应的负对照。若点数少于 1024 或条件数超过
`1e8`，明确回退 raw center。

当前主结果：

- 固定 reference-point Sim(3) ATE RMSE `0.3745 -> 0.1759 m`（`-53.0%`）；
- RPE translation RMSE `0.1522 -> 0.0809 m`（`-46.8%`）；
- translation-direction mean `14.56° -> 11.35°`；
- translation@10° `33.3% -> 71.4%`；
- 帧 119 relative error `0.1960 -> 0.0965 m`；
- 帧 240 `0.4597 -> 0.2124 m`；
- shuffled ATE `1.4202 m`、direction mean `44.16°`。

所有七帧均成功求解，平均条件数约 `1.72`。GT-K ATE 为 `0.1238 m`，GT-K+R
固定 point-Sim3 ATE 为 `0.0886 m`，但 oracle ray residual 反而高于 predicted K，
所以不能用本帧 ray residual 自监督 focal。

剩余关键失败是 `210->240`：方向误差仅从 `44.22°` 降至 `43.96°`，预测运动
长度从 `0.312 m` 降到 `0.207 m`，仍高于 GT `0.123 m`。`105->119` 则从
`34.33°` 降到 `18.91°`。下一步应以可靠 persistent static-instance map 为跨帧
锚点处理帧 240，而不是继续调 ray trimming。

新输出为 `ray_fit_*.csv` 与 `ray_pose_*.csv`；唯一服务器命令已更新到
`streaming_couping/commands.txt`。本阶段仍不运行 SAM3；只有 ray-center 被证明
能改善固定 point-Sim3 下的 ATE/RPE 后，才接可靠多实例点云。
