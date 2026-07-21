# StreamVGGT 相机位姿与点图诊断

## 目的

在引入实例点云约束前，先建立 frozen StreamVGGT 的 raw geometry baseline，
回答三个问题：

1. camera head 与 point head 是否处于一致的预测世界坐标系；
2. 相对 reference 帧的相机轨迹是否随时间漂移；
3. 后续帧 pointmap 误差主要与 pose、intrinsics 还是局部几何误差相关。

本诊断不加载 SAM3，不读取实例 mask，也不使用实例约束。raw 表保持原始
StreamVGGT 输出；新增 ray-center 分支不修改 pointmap、rotation 或 intrinsics，
只由 pointmap 重新估计 camera center。GT 只用于 evaluation 和显式 oracle。

## 坐标约定

ScanNet++ manifest 保存 COLMAP world-to-camera：

```text
X_camera = R_world_to_camera @ X_world + t
camera_center_world = -R_world_to_camera.T @ t
```

StreamVGGT 的 camera token 通过 `pose_encoding_to_extri_intri` 解码为相同形式的
3×4 world-to-camera。诊断先把旋转投影到最近的 SO(3)，并把投影前的 determinant、
orthogonality error 和投影改变量单独写入 `pose_rotation_quality.csv`。

单目 StreamVGGT 的世界坐标存在 Sim(3) gauge，不能直接比较原始 translation。

## 三种轨迹对齐

### `reference_point_sim3`

主评估。只使用 reference 帧全场景逐像素对应点拟合一次 robust Sim(3)，然后对
全部后续 pointmap 和 camera pose 保持固定。

该模式检查：

- StreamVGGT point head 的世界坐标漂移；
- camera head 是否与 point head 使用相同 gauge；
- 不借助后续 GT 时的绝对轨迹误差。

### `reference_pose_point_scale`

使用 reference 帧相机中心和方向固定 rotation/translation，尺度沿用
`reference_point_sim3`。reference pose 按定义应接近零误差，后续误差主要反映
相对轨迹漂移。

### `trajectory_sim3`

使用全部选中帧的 GT camera centers 拟合 Sim(3)。这是标准但乐观的 gauge
诊断，只用来判断轨迹形状是否正确，不能作为后续实例约束方法的主结果。

## 输出

- `pose_summary.csv`
  - ATE RMSE；
  - absolute translation/rotation error；
  - adjacent-frame RPE；
  - 三种 alignment 的 Sim(3)。
- `pose_frame_metrics.csv`
  - raw/aligned/GT camera center；
  - 每帧 translation 与 rotation error。
- `pose_rpe.csv`
  - 相邻选中帧的 relative translation/rotation error；
  - predicted/GT motion magnitude。
- `pose_pair_metrics.csv`、`pose_pair_summary.csv`
  - 与 StreamVGGT 官方 pose benchmark 一致的所有帧对相对旋转误差；
  - 尺度无关、正负号模糊的平移方向误差与 5/10/30 度 accuracy。
- `pose_rotation_quality.csv`
  - camera head rotation matrix 的 SO(3) 合法性。
- `pointmap_frame_metrics.csv`
  - 固定 reference Sim(3) 后逐像素配对距离的 mean/median/RMSE/P90。
- `pointmap_summary.csv`
  - reference 与 non-reference pointmap 误差汇总。
- `intrinsics_frame_metrics.csv`、`intrinsics_summary.csv`
  - 将 manifest intrinsics 严格变换到 StreamVGGT crop 后与预测内参比较。
- `ray_fit_frame_metrics.csv`、`ray_fit_summary.csv`
  - 每帧求解状态、条件数、保留点比例、camera-center 改变量；
  - 全部点与 trimmed inlier 的射线残差；
  - native pointmap 单位和固定 reference Sim(3) 后的米制残差。
  - 可直接复用的 repaired world-to-camera rotation 与 translation。
- `ray_pose_summary.csv`、`ray_pose_frame_metrics.csv`、`ray_pose_rpe.csv`
  - raw 与六个 ray-center 分支的 ATE、逐帧误差和 adjacent RPE；
  - `reference_point_sim3` 是主比较，所有分支共用同一次固定对齐；
  - 每个分支另给 reference-pose 对齐结果，只诊断相对漂移。
- `ray_pose_pair_metrics.csv`、`ray_pose_pair_summary.csv`
  - ray-center 修复后的官方风格 all-pairs rotation/translation direction。
- `metadata.json`
  - 数据边界、坐标约定、reference Sim(3) 和输出清单。

## Pointmap-consistent ray-center 修复

StreamVGGT 的 point head 与 camera head 从共享 token 独立解码。对处理后像素
\((u_i,v_i)\)，由固定内参和 camera-to-world rotation 得到单位世界射线
\(d_i\)，point head 给出对应世界点 \(X_i\)。理想情况下存在深度
\(\lambda_i\) 使：

```text
X_i = C + lambda_i * d_i
```

因此只需求解相机中心：

```text
C* = argmin_C sum_i w_i ||(I - d_i d_i^T)(X_i - C)||^2
```

这是 3×3 加权线性法方程。服务器结果表明 confidence gate 后的全部射线优于
80% residual trimming，因此可部署主分支使用 predicted `K/R`、归一化 point
confidence 和全部通过 gate 的射线；trimming 只保留为稳健性消融。若有效点少于
1024，或法方程条件数超过 `1e8`，该帧明确回退 raw camera center。最后保持
world-to-camera rotation 不变，仅更新：

```text
t_repaired = -R_world_to_camera @ C*
```

拟合发生在 pointmap 原生 gauge，不需要 GT Sim(3)。固定 reference Sim(3) 仅在
输出米制评估指标时使用。

### 一次性分支

| mode | K | R | trimming | 用途 |
|---|---|---|---|---|
| `raw_camera_head` | predicted | predicted | — | 原始基线 |
| `ray_predicted_k_all` | predicted | predicted | 否 | 可部署主方法 |
| `ray_predicted_k_trimmed` | predicted | predicted | 是 | trimming 消融 |
| `ray_gt_k_trimmed` | GT processed | predicted | 是 | 内参 oracle |
| `ray_gt_r_trimmed` | predicted | GT gauge-transformed | 是 | 旋转 oracle |
| `ray_gt_k_gt_r_trimmed` | GT processed | GT gauge-transformed | 是 | pointmap 上限 |
| `ray_shuffled_pointmap_trimmed` | predicted | predicted | 是 | 像素—点对应负对照 |

GT rotation 会通过 evaluation-only reference point Sim(3) 的 rotation 变换回
pointmap gauge。shuffled 分支把每帧 pointmap 水平循环移动三分之一宽度，保持点
分布但破坏像素与世界点对应。

## 当前单场景结果

场景 `00a231a370`、帧 `90 105 119 130 140 210 240`：

- 固定 reference-point Sim(3) 下，主分支 ATE RMSE
  `0.3745 -> 0.1759 m`（`-53.0%`）；
- adjacent RPE translation RMSE `0.1522 -> 0.0809 m`（`-46.8%`）；
- all-pairs translation-direction mean `14.56° -> 11.35°`；
- translation@10° `33.3% -> 71.4%`；
- 帧 119 reference-pose-aligned error `0.1960 -> 0.0965 m`；
- 帧 240 `0.4597 -> 0.2124 m`；
- shuffled pointmap ATE 为 `1.4202 m`、方向均值 `44.16°`。

全点版在全部非 reference 帧的绝对误差上均优于 trimmed。GT-K oracle 的 ATE 为
`0.1238 m`，GT-K+R 的固定 point-Sim3 ATE 为 `0.0886 m`。但 predicted K 的
内部 ray residual 低于 GT K，说明 ray residual 只能衡量两个 head 的内部一致性，
不能单独监督真实 focal/pose。

局部失败并未全部消失：`105->119` 的方向误差从 `34.33°` 降至 `18.91°`，而
`210->240` 仍为 `43.96°`。后者与帧 240 最大 pointmap RMSE 同时出现，是下一步
persistent static-instance geometry 需要处理的边界。

## 判读

1. `reference_point_sim3` 的 reference pose 误差很大：
   - 优先检查 camera head 与 point head gauge、world-to-camera 方向或坐标轴约定。
2. `reference_pose_point_scale` 的误差随帧间隔增长：
   - 说明存在相对相机漂移，才进入实例点云位姿约束研究。
3. `trajectory_sim3` 明显优于前两种：
   - 轨迹形状可能正确，主要问题是 gauge 或累积尺度/方向偏移。
4. trajectory Sim(3) 后 ATE/RPE 仍高：
   - 相机运动形状本身错误，不应先用单实例 ICP 掩盖。
5. pose 误差与 pointmap RMSE 同时增长：
   - 可以研究多静态实例共享的 pose correction。
6. pose 较准而 pointmap 误差高：
   - 优先研究 depth/point head 与点云融合，而不是改 pose。
7. predicted intrinsics 明显偏离处理后 GT：
   - 先排除内参误差，再解释 pose 或重投影残差。
8. `ray_predicted_k_all` 的固定 point-Sim3 ATE/RPE 优于 raw：
   - point head 可以作为 camera-head translation 的无训练后处理约束。
9. all 优于 trimmed，且 shuffled 明显恶化：
   - 当前 confidence gate 后不需要额外 hard trimming，改善来自正确的像素—点
     射线对应，而非任意中心重拟合。
10. GT-K 明显优于 predicted-K：
    - 后续加入场景级 focal calibration；否则保持 predicted K。
11. ray residual 降低但 pose ATE 不降：
    - 两个 head 在错误的内部几何上自洽，不能把 ray fit 宣称为位姿修复。

本诊断阶段已经结束，独立 CLI 已删除；结果保留在本文档，仍被当前实验使用的
ray-center、ATE/RPE 和 all-pairs 算子已整理到 `src/pose_evaluation.py`。当前
第三版实验命令见 `../commands.txt`。
