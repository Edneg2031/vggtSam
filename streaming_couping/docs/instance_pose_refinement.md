# 多实例 pointmap / 相机位姿修复第二版

## 研究边界

当前链路为：

```text
geometry -> SAM3 tracking recovery（已验证）
tracking -> persistent static-instance maps
instance maps -> 一个整帧共享 pointmap translation
corrected pointmap -> all-point ray-center camera translation
```

相机和整帧 pointmap 从不按实例分别修改。reference 后 GT 只用于指标和名称中明确
标出的 `gt_point_translation_oracle`。

## 第一版负结果

第一版把共识平移同时写入所有参与实例的 object map，并使用较宽的
correspondence/consensus gate。服务器结果：

| mode | fixed-reference ATE | adjacent RPE translation RMSE |
|---|---:|---:|
| `ray_only` | 0.175915 m | 0.080938 m |
| V1 causal a100 | 0.181835 m | 0.090999 m |
| V1 causal a025 | 0.176317 m | 0.080894 m |
| GT translation oracle | 0.128355 m | 0.067590 m |

V1 a100 相比 `ray_only` 的 ATE 恶化 3.37%，RPE 恶化 12.43%，因此第一版不能
作为主方法。它仍在第二版同次实验中以 `v1_shared_map_a100` 复现，便于排除代码
版本差异。

主要失败原因：

1. shared correction 写进每个实例地图，掩盖实例间 proposal 差异并污染后续配准；
2. correspondence ratio 0.15、consensus distance 0.05 过宽；
3. 帧 130 有修正、140 归零，令 130→140 direction error 从 29.63° 增至 75.78°；
4. bed 的恢复 mask 很好，但跨视角表面与历史地图几乎无几何重叠，无法提供 ICP
   proposal。这说明 tracking recovery 成功不等于该实例一定能约束 pose。

V1 a050 的 non-reference pointmap RMSE 从 0.132935 m 降到 0.130259 m，证明实例
几何存在弱有效信号；但它没有转化成 pose 改善。GT translation oracle 的
non-reference pointmap RMSE 为 0.115600 m，说明仍有可研究空间。

## 第二版计算图

每个实例仍用 translation-only trimmed nearest-neighbor ICP：

```text
j(i) = NN(P_t^k[i] + delta, O_<t^k)
delta <- delta + coordinate_median(
  O_<t^k[j(i)] - (P_t^k[i] + delta)
)
```

严格 proposal gate：

```text
min points              = 128
min fitness             = 0.25
max ICP RMSE             = 0.03 native
correspondence distance  = max(0.02, 0.05 * object scale)
max translation norm     = 0.15 native
```

至少两个 proposal 围绕最终 coordinate-wise median 的最大残差不超过 0.02，
才产生多实例共享平移。

第二版将地图与相机职责分开：

```text
object map k update:  P_t^k + delta_t^k
whole-frame update:   X'_t = X_t + alpha * Delta_t
```

即每个通过严格 gate 的实例只用自己的 `delta_t^k` 更新自己的地图；相机仍只使用
一个 `Delta_t`。这不恢复“每实例独立改相机”的旧方案。

## 短期连续机制

多实例共识失败时，只有以下条件全部成立，单实例 proposal 才能产生整帧修正：

- 当前恰好一个 proposal 通过严格 gate；
- 距最近一次多实例共识的源帧间隔不超过 15；
- proposal 与最近共享平移的距离不超过 0.02；
- tracker score 不低于 map update score gate。

接受后可以更新平移值，但 15 帧窗口始终锚定最近一次多实例共识，不能靠连续单实例
carry 无限向后滚动。平移使用：

```text
Delta_t = 0.5 * previous_Delta + 0.5 * current_delta
```

若多个 proposal 通过但互相冲突，carry 被明确禁止且历史连续状态立即清空。长间隔
同样清空，因而不会把帧 140 的状态无条件带到 210/240。

## 同次消融

| mode | 唯一变量 |
|---|---|
| `ray_only` | 无实例修正 |
| `v1_shared_map_a100` | 第一版复现 |
| `v2_strict_shared_map_no_carry_a100` | 只收紧 gate |
| `v2_strict_per_instance_no_carry_a100` | 再加入每实例地图更新 |
| `v2_strict_per_instance_short_carry_a025/a050/a100` | 再加入短期连续，alpha 消融 |
| `v2_strict_reference_no_carry_a100` | 只用 reference map |
| `v2_strict_shuffled_short_carry_a100` | shuffled-ID 负对照 |
| `gt_point_translation_oracle` | evaluation-only 上限 |

当前主候选是 `v2_strict_per_instance_short_carry_a100`，在服务器结果回来前只称
candidate，不称已验证方法。

## 输出与判读

优先检查：

1. `instance_correction_events.csv`
   - `correction_source`
   - `temporal_frame_gap`
   - `temporal_proposal_distance_native`
   - `map_update_instance_ids`
2. `instance_icp_diagnostics.csv`
   - `icp_rmse_native`
   - `consensus_participant`
   - `temporal_participant`
   - `map_updated`
   - 每实例 `map_update_translation_native_*`
3. `instance_pose_summary.csv`、`instance_pose_rpe.csv`
4. `instance_pose_pair_summary.csv` 与 `instance_pose_pair_metrics.csv`
5. `instance_pointmap_summary.csv`、`instance_pointmap_frame_metrics.csv`

重点 pairs：105→119、119→130、130→140、210→240。

PLY 保留：

```text
instance_<id>/pointclouds/ray_only/
instance_<id>/pointclouds/v2_strict_per_instance_short_carry_a100/
```

唯一服务器入口见 [`../commands.txt`](../commands.txt)。
