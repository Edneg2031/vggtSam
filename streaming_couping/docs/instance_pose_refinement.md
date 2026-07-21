# 多实例 pointmap / 相机位姿修复最终 V3

## 研究边界

当前链路为：

```text
geometry -> SAM3 tracking recovery（已验证）
tracking -> persistent static-instance maps
instance maps -> 一个整帧共享 pointmap translation
corrected pointmap -> all-point ray-center camera translation
```

相机和整帧 pointmap 从不按实例分别修改。reference 后 GT 只用于指标，不参与
proposal、共识、temporal filtering、map 写回或 ray fit。实例 PLY 始终保留。

## 已完成的 V1/V2 结论

真正的 baseline 是已经修复过 camera center 的 `ray_only`，不是 raw camera head。

| mode | fixed-reference ATE | adjacent RPE translation RMSE | all-pairs direction mean |
|---|---:|---:|---:|
| `ray_only` | 0.175915 m | 0.080938 m | 11.345° |
| V1 shared-map a100 | 0.181835 m | 0.090999 m | 14.393° |
| V2 per-instance a050 | 0.176086 m | 0.078713 m | 11.820° |
| GT translation oracle | 0.128355 m | 0.067590 m | 9.588° |

V1 明显退化。V2 a050 将 adjacent RPE 改善 2.75%，但 official-style all-pairs
方向均值恶化 4.18%，translation@10° 从 71.43% 降到 66.67%，因此也不能作为
最终方法。GT oracle 的 all-pairs 均值改善 15.48%，说明 pointmap translation
存在可利用信号，当前瓶颈是 proposal 验证而不是方向本身无效。

V2 的具体失败原因：

1. ICP fitness/RMSE 很好也可能落入错误的局部匹配盆地；
2. 帧 130 cabinet/wardrobe proposal disagreement 为 0.0641，大于 0.02；
3. 帧 240 disagreement 为 0.0468；
4. V2 会把所有局部 accepted proposal 写回各自 map，即使该实例没有参与共识；
5. V2 在多 proposal 冲突时清空 temporal state，导致 130/140 无法连续利用可信的
   wardrobe proposal；
6. V2 short-carry 与 no-carry 结果相同，说明旧 carry 实际从未触发。

## V3 proposal 与共识

每实例仍使用 translation-only trimmed nearest-neighbor ICP。严格 gate 不变：

```text
min points              = 128
min fitness             = 0.25
max ICP RMSE             = 0.03 native
correspondence distance  = max(0.02, 0.05 * object scale)
max translation norm     = 0.15 native
consensus max residual   = 0.02 native
```

tracker score 低于 map-update score gate 的 proposal 不参与 V3 共识或 temporal
筛选。至少两个 eligible proposal 围绕最终 coordinate-wise median 的最大残差不
超过 0.02，才产生普通多实例共识。

## V3 temporal conflict filtering

当普通共识失败时，不再因为“有多个 proposal”直接清空历史。将每个 eligible
proposal 与上一轮共享平移比较：

```text
temporal_distance(k) = ||delta_t^k - previous_Delta||
```

只有以下条件全部成立才接受：

- 当前恰好一个 temporal inlier，距离不超过 0.02；
- 与上一轮有效修正的源帧 gap 不超过 15；
- 连续 single-instance carry 尚未达到 2 次上限。

接受时：

```text
Delta_t = 0.5 * previous_Delta + 0.5 * delta_t^k
X'_t = X_t + 0.5 * Delta_t
```

与 V2 不同，carry 接受后 temporal frame 会前移，因此允许
`119 consensus -> 130 carry #1 -> 140 carry #2`。连续次数达到 2、长 gap、没有
唯一 inlier 或中间拒绝都会清空状态；210 的长间隔不能继承到 240。

## 受控 object-map 写回

V3 主分支只允许最终通过相机决策验证的实例更新自己的 map：

```text
multi consensus participant
or unique validated temporal participant
    -> O_t^k = merge(O_<t^k, P_t^k + delta_t^k)
```

被共识排除的 proposal、冲突中的 temporal outlier 和最终拒绝帧一律不写回。
map 使用各实例未缩放的 `delta_t^k`；整帧 pointmap/相机始终只接收一个共享且乘
`alpha=0.5` 的 translation，不恢复每实例独立改相机。

## V3 最终结果

| mode | ATE | adjacent RPE | all-pairs mean | all-pairs median |
|---|---:|---:|---:|---:|
| `ray_only` | 0.175915 m | 0.080938 m | 11.345° | 7.383° |
| V2 a050 | 0.176086 m | 0.078713 m | 11.820° | 7.385° |
| final V3 | **0.173651 m** | 0.080995 m | **10.780°** | **6.942°** |

V3 相对 `ray_only` 的 ATE 改善 1.29%，all-pairs mean 改善 4.99%，median
改善 5.98%；RPE 仅变化 +0.07%，视为持平。translation@5° 从 28.57% 提升到
33.33%，@10°/@30° 不下降。shuffled 负对照与 baseline 完全一致。

状态机按设计运行：130 排除 cabinet、wardrobe carry #1；140 wardrobe carry #2；
210 因 gap=70 和 carry 上限 reset；240 无历史时拒绝冲突。validated 写回只更新
最终 participating IDs。

## 当前保留的运行分支

| mode | 用途 |
|---|---|
| `ray_only` | 最小 sanity baseline |
| `v3_temporal_validated_a050` | 最终 V3 方法 |

V1/V2、consensus-only、unvalidated-map、shuffled、GT oracle 和 alpha sweep 已完成
研究任务，从运行时代码删除；历史结果保留在本文和 Git 历史。

## 输出与判读

优先检查：

1. `instance_correction_events.csv`
   - `correction_source`
   - `temporal_reference_frame_index`
   - `temporal_inlier_instance_ids`
   - `temporal_carry_count_before/after`
   - `map_update_instance_ids`
2. `instance_icp_diagnostics.csv`
   - `proposal_score_eligible`
   - `temporal_distance_native`
   - `temporal_inlier`
   - `proposal_validated`
   - `map_updated`
3. `instance_pose_summary.csv`、`instance_pose_rpe.csv`
4. `instance_pose_pair_summary.csv` 与 `instance_pose_pair_metrics.csv`
5. `instance_pointmap_summary.csv`、`instance_pointmap_frame_metrics.csv`

重点确认帧 130 是否排除 cabinet 并保留 wardrobe、140 是否成为 carry #2、210
是否因长 gap reset、240 是否在没有有效 temporal history 时拒绝。最终判断必须同时
看 ATE、adjacent RPE 和 all-pairs，不能只凭 pointmap RMSE 或相邻帧指标。

PLY 输出包含：

```text
instance_<id>/pointclouds/ray_only/
instance_<id>/pointclouds/v3_temporal_validated_a050/
```

唯一服务器入口见 [`../commands.txt`](../commands.txt)。
