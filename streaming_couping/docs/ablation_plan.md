# 一次性双向耦合消融设计

## 1. 本轮要回答的问题

固定场景 `00a231a370` 和时序帧
`90 105 119 130 140 210 240`，一次运行回答五个问题：

1. SAM3 的全图文本候选中是否存在正确实例？
2. 时序对齐的 StreamVGGT 几何是否能从同类候选中选对实例？
3. 完全相同的恢复 mask 写入原 `obj_id` memory 后，是否改善未来帧？
4. 可靠 SAM3 mask 扩充 persistent 3D object map 后，是否比只用 reference
   几何更利于后续恢复？
5. tracker-conditioned object map 的纯度和完整度是否提高？

这是一组机制验证，不是最终论文规模 benchmark。单场景结果可以决定代码路线，
但不能单独支撑泛化结论。

## 2. 测试实例

| 实例 | 类别 | 角色 |
|---|---|---|
| 37 | cabinet | SAM3 易例；检查完整方法是否误伤 |
| 68 | wardrobe | SAM3 易例；检查完整方法是否误伤 |
| 54 | bed | 高置信低 IoU 压力例；检查 geometry-disagreement gate |

bed 不再从实验中删除。它不会与小物体共用面积塌缩规则；当前 gate 使用的是
SAM3 score、对齐几何覆盖率和候选支持覆盖率。

## 3. 两种事件策略

### natural_joint_gate

可部署策略。以下任一条件成立时 tracker 被视为弱：

- mask 为空；
- SAM3 score 低于阈值；
- mask 虽非空且 score 高，但覆盖的对齐 3D 支持低于阈值。

恢复还必须满足：历史 3D map 在当前 pointmap 中得到有效支持，并且选中的完整
SAM3 candidate 覆盖足够多的几何支持点。

### scheduled_probe

机制探针。优先在序列位置 4，即帧 140，绕过“tracker 是否弱”这一项，但仍保留
几何有效性与候选覆盖检查。这样即使 37/68 太容易，也能测试 memory writeback
是否安全。

如果帧 140 没有可执行候选，程序会在不读取 GT 的情况下选择最早可执行的后续
帧。实际帧与 fallback 标志写入 `summary.csv`。

## 4. 七个分支

| mode | 改变量 | 回答的问题 |
|---|---|---|
| `original` | 原始 SAM3 | 2D tracking baseline |
| `geometry_recovery_no_memory` | 只替换恢复帧，不写 memory | 当前帧 mask 的直接收益 |
| `reference_geometry_same_id_memory` | 只用 reference 对象点选候选并写回 | 没有 tracking→geometry 的对照 |
| `geometry_recovery_same_id_memory` | 可靠历史 mask 扩图、几何选候选、same-ID 写回 | 完整双向主分支 |
| `shuffled_geometry_same_id_memory` | 打乱非 reference 几何，其余不变 | 时序对齐几何的负对照 |
| `oracle_candidate_same_id_memory` | GT 只在候选集合中选最优 mask | 候选选择上限 |
| `oracle_mask_same_id_memory` | 恢复帧直接写 GT mask | SAM3 memory writeback 上限 |

`geometry_recovery_no_memory` 与
`geometry_recovery_same_id_memory` 使用逐像素完全相同的恢复 mask；两者恢复后的
差异只能来自 SAM3 memory。

## 5. tracking → geometry 消融

每个 tracking 分支都生成三种对象地图：

- `all_frames`：所有预测 mask 均写图；
- `score_gate`：只保留 SAM3 score 通过的帧；
- `joint_gate`：SAM3 score 和历史 3D 几何覆盖同时通过。

另有两个对照：

- `gt_mask_oracle`：对象地图上限；
- `time_shuffled_original_masks`：打乱 mask 与 pointmap 的时间对应。

StreamVGGT pointmap 只通过 reference 帧全场景点拟合一次固定 Sim(3) 到 ScanNet++
GT 坐标，此变换对全部帧和分支保持不变。GT pointmap 仅用于评估：

- object-map precision / recall / F-score @ 5 cm；
- object-map precision / recall / F-score @ 10 cm；
- 双向 Chamfer-L1；
- 写图帧数和点数。

不做逐帧 GT 对齐，不做 ICP，也不利用 GT 修正相机或对象地图。

## 6. 一次运行内的阈值诊断

程序对全部六个 post-reference 帧缓存全图文本候选，并在不重跑模型的情况下输出
27 组阈值组合：

- tracker score：`0.30 / 0.50 / 0.70`
- tracker geometry coverage：`0.10 / 0.25 / 0.50`
- selected candidate support coverage：`0.25 / 0.50 / 0.75`

`threshold_sweep.csv` 中的 GT 只用于 post-hoc 统计 gate precision/recall，
不能把该文件中最优阈值当成无偏测试结果。它的用途是选择下一阶段固定阈值。

## 7. 判读顺序

1. 看 `candidate_screening.csv`：
   - `oracle_candidate_gt_iou` 低：候选生成失败；
   - oracle 高、`selected_candidate_gt_iou` 低：几何选择失败；
   - 两者都高：可以判断 memory。
2. 看 scheduled probe 的 `summary.csv`：
   - full memory 对 no-memory 的 `post_recovery_iou` 增益衡量写回；
   - aligned 对 shuffled 的差异衡量真实几何作用；
   - full 对 reference-only 的差异衡量 tracking→geometry 扩图作用。
3. 看 natural gate：
   - 54 应能识别高置信跟错；
   - 37/68 不应被错误恢复或显著掉点。
4. 看 `map_quality.csv`：
   - `joint_gate` 应比 `all_frames` 有更高 precision/F-score；
   - 完整分支应比 original/reference-only 有更高 recall 或 F-score；
   - GT oracle 应构成合理上限，time-shuffled 应明显更差。

## 8. 本轮通过标准

至少满足：

- scheduled probe 中 aligned geometry 能产生有效候选；
- `geometry_recovery_same_id_memory.post_recovery_iou` 高于
  `geometry_recovery_no_memory.post_recovery_iou`；
- aligned 不低于 shuffled，且 37/68 不被明显误伤；
- candidate oracle 与 GT-mask oracle 能指出剩余瓶颈；
- map 的 joint gate 相比 all-frames 提升纯度或 F-score。

若 exact GT-mask writeback 仍不能改善未来帧，说明瓶颈在 SAM3 memory/传播，
应停止调整几何候选。若 GT-mask 有效而 oracle candidate 无效，问题在候选生成；
若 oracle candidate 有效而 geometry selected 无效，问题才在实例选择。
