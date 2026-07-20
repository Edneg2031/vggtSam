# Causal multi-instance pointmap / camera-pose refinement

## 目标与边界

本实验检验已经可靠的 tracking 和 persistent instance geometry 能否进一步修复
StreamVGGT 的 pointmap translation，并将修复传给 camera pose。它不是新的
SAM3 消融，也不是单实例相机 ICP：

```text
geometry -> tracking recovery（已验证）
tracking -> persistent static-instance maps（已验证实例点云）
instance maps -> one shared frame pointmap translation（当前待验证）
corrected pointmap -> all-point ray-center camera translation（已验证算子）
```

reference 帧 GT mask 是跟踪 prompt 和地图初始化条件。reference 后的 GT mask、
GT pointmap 和 GT pose 不参与 deployable gate、ICP、共识、地图更新或 ray fit。

## 可部署计算图

对静态实例 `k`，reference map 为 `O_0^k`。当前帧 mask 内高置信 point-head 点为
`P_t^k`。translation-only trimmed nearest-neighbor ICP 迭代：

```text
j(i) = NN(P_t^k[i] + delta, O_<t^k)
r_i  = O_<t^k[j(i)] - (P_t^k[i] + delta)
delta <- delta + coordinate_median(r_i in trimmed inliers)
```

ICP 同时尝试 zero 与 robust-centroid 两个初始化；centroid 初值超过最大允许平移时
不会启用。候选先按 inlier fitness、再按 RMSE 选择，最终仍必须经过 magnitude
和跨实例共识 gate。这避免小幅真实平移被 zero-init 最近邻局部极小吞掉，同时用
多实例一致性约束 partial-view centroid 偏差。

每个 proposal 必须满足：

- 当前点和地图点均不少于 128；
- inlier fitness 不低于 0.25；
- proposal norm 不超过 0.15（StreamVGGT native gauge）；
- correspondence distance 为
  `max(0.02, 0.15 * robust_object_scale)`。

通过单实例 gate 的 proposals 再做跨实例共识。至少两个实例距共识中心不超过
0.05 native，随后实例等权的 coordinate-wise median 形成唯一 `Delta_t`。整帧
pointmap 只更新一次：

```text
X'_t = X_t + alpha * Delta_t
```

无共识时 `Delta_t=0`。causal map 只接收参与共识且 tracker score 不低于 0.50
的 observation：

```text
O_t^k = merge(O_<t^k, P_t^k + Delta_t)
```

注意地图始终使用完整 `Delta_t`，alpha 只缩放输出 pointmap correction。因此
alpha 分支共享完全相同的 ICP proposals 和 map history。

最后固定 StreamVGGT predicted `K/R`，用修正后的全点 pointmap 求 camera center：

```text
C'_t = argmin_C sum_i w_i ||(I - d_i d_i^T)(X'_t[i] - C)||^2
t'_t = -R_t C'_t
```

若 ray 点少于 1024 或法方程条件数超过 `1e8`，回退到
`raw_center + applied_translation`。

## 同次消融

| mode | 作用 |
|---|---|
| `raw_camera_head` | StreamVGGT camera-head baseline |
| `ray_only` | 未修 pointmap + 已验证 all-point ray-center |
| `original_causal_a100` | 不做 tracking recovery |
| `recovered_reference_a100` | 只用 reference instance map |
| `recovered_causal_a025/a050/a075/a100` | 可部署 tracking + causal map，固定 alpha 消融 |
| `gt_masks_causal_a100` | reference 后 GT-mask oracle |
| `shuffled_ids_causal_a100` | 当前实例查询错误 ID map 的负对照 |
| `gt_point_translation_oracle` | paired GT pointmap 的 translation-only oracle |

当前 provisional main 是 `recovered_causal_a100`。只有服务器结果支持后才能改为
“已验证主方法”。

## 判读顺序

1. `instance_correction_events.csv`
   - 哪些帧达到至少两个实例共识；
   - 参与的是哪些实例；
   - shared translation、applied translation 和 disagreement。
2. `instance_icp_diagnostics.csv`
   - proposal 是否因点数、fitness 或平移过大被拒绝；
   - 37/68/54 的 proposal 是否同向。
3. `instance_pointmap_summary.csv`
   - `recovered_causal_a100` 是否低于 `ray_only` 的 nonreference RMSE；
   - 帧 240 的 paired RMSE 是否下降。
4. `instance_pose_summary.csv` 与 `instance_pose_rpe.csv`
   - 固定 `reference_point_sim3` ATE；
   - adjacent translation RPE。
5. `instance_pose_pair_summary.csv`
   - all-pairs translation-direction mean / @10°；
   - 逐 pair 文件重点看 `210->240`。

支持方法成立至少需要：

- recovered causal 优于 `ray_only`；
- recovered causal 优于 original causal，证明 recovery 对 geometry 有贡献；
- causal 优于 reference-only，证明历史地图有贡献；
- shuffled IDs 明显更差或被共识 gate 拒绝；
- GT oracles 给出合理、但未被 deployable 分支使用的上限。

如果 `ray_only` 更好，先看是否没有共识；若有共识但方向错误，优先收紧
consensus/fitness 或处理 bed 平面退化，不使用 GT 逐帧挑 alpha。

## 缓存、输出与 PLY

首次运行生成 `tracking_cache.npz`。cache 只缓存 original/recovered masks、
scores、persistent obj IDs 和 tracking summary；它校验实例、帧、输出大小、
mask shape 与 tracking 配置签名。改变实例 ICP/ray 参数不使 cache 失效，改变
tracking 配置会自动重跑 SAM3。

主要 CSV：

- `tracking_summary.csv`
- `instance_correction_events.csv`
- `instance_icp_diagnostics.csv`
- `instance_ray_fit.csv`
- `instance_pose_summary.csv`
- `instance_pose_frame_metrics.csv`
- `instance_pose_rpe.csv`
- `instance_pose_pair_metrics.csv`
- `instance_pose_pair_summary.csv`
- `instance_pointmap_frame_metrics.csv`
- `instance_pointmap_summary.csv`

实例点云保留在：

```text
instance_<id>/pointclouds/ray_only/
instance_<id>/pointclouds/recovered_causal_a100/
```

唯一服务器命令见 [`../commands.txt`](../commands.txt)。
