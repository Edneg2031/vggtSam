# Streaming coupling method

## 目标

研究冻结的 SAM3 与 StreamVGGT 如何形成可解释的双向闭环：

```text
geometry helps tracking
tracking builds persistent instance geometry
multiple static instances help pointmap translation
pointmap helps camera translation
```

系统不训练 token fusion，也不让单个实例独立修改相机。reference 帧 GT mask 是
允许的初始化条件；reference 后 GT 只用于评估和明确命名的 oracle。

## 1. Geometry helps tracking

### 1.1 Persistent object support

reference mask 内的 StreamVGGT point-head 点初始化实例地图。历史点投影到当前帧，
再与当前 pointmap 做 3D 支持检验，形成 `supported_mask`。

SAM3 mask 即使 score 很高，只要覆盖的 geometry support 太低，也被视为高置信
跟错：

```text
tracker_geometry_coverage < 0.50
```

### 1.2 Full-mask recovery and identity

触发后 SAM3 在全图产生同语义候选。几何只负责在候选间确认历史实例，不用局部
point/box 截断完整边界。候选的 support coverage 至少为 0.25 才接受。

被选中的完整 mask 写回原 SAM3 `obj_id` memory，后续传播继续保持同一实例。

已验证的 bed 压力测试：

| branch | frame 119 IoU | later-four-frame mean IoU |
|---|---:|---:|
| original | 0.0207 | 0.0002 |
| geometry recovery, no memory | 0.9323 | 0.0002 |
| geometry recovery + same-ID memory | 0.9323 | 0.7763 |

该阶段已经结束；当前只保留 `tracking_recovery.py` 的可部署路径。

## 2. Tracking builds persistent instance maps

可靠 recovered mask 转到 StreamVGGT grid，在 mask 内选择高置信 point-head 点。
reference observation 强制使用 reference GT prompt mask，后续只使用预测 tracking。

每个实例维护独立 object map `O^k`。实例 PLY 始终导出，便于观察表面覆盖、漂移与
污染，不因相机实验精简而删除。

## 3. Ray-center baseline

固定 StreamVGGT predicted intrinsics `K` 和 rotation `R`。像素射线在世界坐标为
`d_i`，pointmap 点为 `X_i`，相机中心由：

```text
argmin_C sum_i w_i ||(I - d_i d_i^T)(X_i - C)||^2
t = -R C
```

求解。该算子已验证：

- fixed-reference ATE：0.3745 → 0.1759 m；
- adjacent RPE translation RMSE：0.1522 → 0.0809 m；
- all-pairs direction mean：14.56° → 11.35°；
- translation@10°：33.3% → 71.4%。

因此实例方法必须与 `ray_only` 比，而不是只与 raw camera head 比。

## 4. V1 instance correction and failure

V1 对每个静态实例做 translation-only trimmed NN ICP，再取跨实例 median 作为唯一
整帧修正：

```text
X'_t = X_t + alpha * Delta_t
```

但 V1 同时用 `Delta_t` 更新所有参与实例地图。服务器结果中 a100 的 ATE/RPE 均
劣于 `ray_only`；a025 仅打平。主要问题是地图污染、gate 过宽和稀疏修正造成
130→140 的时间跳变。

该分支已经由服务器实验回答，不再进入当前运行，不能称为有效 pose 方法。

## 5. V2 result and V3 instance correction

### 5.1 Strict proposal gate

每个实例 proposal 仍来自 translation-only ICP，但使用：

```text
min points              128
min fitness             0.25
max RMSE                0.03 native
correspondence distance max(0.02, 0.05 * object scale)
max translation         0.15 native
```

多实例共识的最终 robust-center 最大残差必须不超过 0.02 native。

### 5.2 V2 final result

V2 让每个局部 accepted proposal 用自己的平移更新自己的 map，并加入锚定最近
多实例共识的 short carry。a050 的 adjacent RPE 改善 2.75%，但 all-pairs direction
mean 恶化 4.18%，translation@10° 也下降，因此 V2 只保留为复现对照。

根因是局部 ICP accepted 不等于跨实例或时间上可信。被共识排除以及互相冲突的
proposal 仍然写入 map，且旧 carry 在冲突时直接清空，实际从未触发。

### 5.3 Validated map writeback

V3 只有最终参与普通共识或 temporal carry 的实例才能写回：

```text
O_t^k = merge(O_<t^k, P_t^k + delta_t^k)
```

其余 accepted-but-unvalidated proposal 不更新 map。整帧 pointmap 和相机仍只使用
一个共享平移：

```text
X'_t = X_t + 0.5 * Delta_t
```

### 5.4 Temporal conflict filtering

普通共识失败后，V3 计算所有 eligible proposal 到上一轮共享平移的距离。恰好
一个 proposal 在 0.02 内、相邻有效源帧 gap ≤ 15 且连续 carry 少于 2 次时，用
0.5/0.5 blend 接受。carry 后 temporal frame 前移，因此最多形成两步短链；长 gap、
没有唯一 inlier 或中间拒绝立即 reset。

最终 V3 在固定 7 帧压力序列上将 ATE `0.175915 -> 0.173651 m`，all-pairs
translation direction mean `11.345° -> 10.780°`、median `7.383° -> 6.942°`，
adjacent RPE 保持在 `0.08094 m` 左右。V3 因此冻结为当前主方法。

详细消融与输出字段见
[`instance_pose_refinement.md`](instance_pose_refinement.md)。

## 6. Evaluation protocol

### Pose

- fixed reference-point Sim(3) ATE；
- adjacent RPE translation/rotation；
- StreamVGGT official-style all-pairs rotation 与 sign-ambiguous translation
  direction；
- 重点检查 105→119、119→130、130→140、210→240。

### Pointmap

只用 reference 帧 paired full-scene points 拟合一次 Sim(3)，后续固定。报告逐帧
paired distance mean/median/RMSE/p90，以及 non-reference 汇总。

### 当前运行分支

- `ray_only`
- `v3_temporal_validated_a050`

历史消融已完成并从运行时代码删除，结果保留在
[`instance_pose_refinement.md`](instance_pose_refinement.md)。

## 7. 当前服务器入口

从仓库根目录运行：

```bash
zsh streaming_couping/commands.txt
```

同一命令复用 tracking cache，只运行 `ray_only` 与最终 V3，并保留实例 PLY。

## 8. Learned persistent-instance pose adapter

该实验分支保留上述显式 V3，不把 fused token写入 SAM3。可靠 recovered mask和
persistent ID生成因果 instance memory；current/history center、covariance、ICP
quality和二者残差经 MLP 编码后，只在 CameraHead前更新最终 hidden camera token。

Camera residual使用 zero-initialized projection，module-off和初始化必须逐元素恢复
raw StreamVGGT。geometry-only、SAM-only、combined和all-token构成结构消融；zero、
shuffled ID和shuffled time均在同一个 trained checkpoint推理时测试。详细设计与
命令见 [`instance_token_pose.md`](instance_token_pose.md)。
