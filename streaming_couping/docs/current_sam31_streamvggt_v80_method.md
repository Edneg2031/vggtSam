# SAM3.1 辅助 StreamVGGT 的当前方法（V8.0/V8.1）

> 当前实现更新于 2026-08-04。V8 不再用单一 pose loss 同时训练 matcher 和 pose head，
> 而是先证明 SAM 是否改善跨视角对应，再验证冻结对应关系能否改善未来帧位姿。

> 历史实验、实际结果、失败结论和后续实验准入规则统一记录在
> [SAM3.1 × StreamVGGT 实验验证账本](sam31_streamvggt_experiment_ledger.md)。设计新分支前必须先
> 检查该账本；本文件描述实现方法，不覆盖账本中的实际 pass/fail 结论。

## 1. V8 要回答的问题

V7.3/V7.4 已证明不同 fusion head 都可以把训练帧 pose loss 拟合到接近零，但没有证明
SAM3.1 局部 token 学到了可以外推到未来帧的对应关系。V8 将证据链拆开：

```text
SAM3.1 local token
  → held-out correspondence 更准确
  → 冻结 p_ij 后，受限 pose residual 更准确
  → SAM-off / wrong-ID / shuffle-time 后退化
```

只有这三步同时成立，CSV 才会将 `fold_sam_causal_pass` 标为 1。

## 2. 推理数据流

SAM3.1 和 StreamVGGT 均保持冻结：

```text
SAM3.1 forward-only tracking
  ├─ 动态 persistent slot / mask / confidence
  └─ mask 内 detector_fpn2 局部 descriptor

StreamVGGT
  ├─ frozen L0 pose
  └─ mask 内 [world xyz, uv, log depth, confidence] geometry token

当前实例 token + 上一次可靠历史观测
  → geometry affinity / SAM affinity
  → masked p_ij
  → transport 历史 geometry Value
  → point evidence → instance pool → multi-instance pool
  → evidence-only bounded SE(3) residual
  → refined pose
```

物体可在 frame 90 之后出生。出生帧只写 memory，下一次可靠观测才允许参与修正。
SAM Key、geometry Key/Value 与训练期 GT correspondence target 使用同一个 causal write
事件；如果当前 geometry 写入但 SAM descriptor 缺失，SAM memory 会同步写成 invalid，不会继续
保留一个来自更老帧、与 geometry Value 错位的 Key。

## 3. 两阶段训练

### 3.1 阶段 A：显式 correspondence 监督

V8 不把 StreamVGGT 预测 world point 再乘 GT pose。缓存中的 `target_world_points` 已是
mesh-rasterized、统一 GT 世界坐标系下的 pointmap。根据每个 geometry token 的归一化 UV，
直接采样：

```math
X^{gt}_{t,i}=P^{gt}_t(v_i,u_i)
```

对 current token 与 matcher 实际读取的 causal history token 计算：

```math
D_{ij}=\|X^{gt}_{t,i}-X^{gt}_{h,j}\|_2
```

只监督满足以下条件的 query：

- current/history GT 点均有限；
- 属于同一 persistent slot；
- 最近邻距离不超过 `0.10 m`；
- 默认要求 mutual nearest neighbour。

没有可靠历史对应的 query 被排除，不会被迫从错误 Key 中选一个。半径内候选形成温度为
`0.025 m` 的软目标分布 `q_ij`，训练损失为 `KL(q || p)`。该阶段只训练 SAM/geometry
affinity encoder 和 Q/K projection，不训练 pose head。

### 3.2 阶段 B：冻结匹配，再训练 pose

阶段 A 结束后，所有会改变 `p_ij` 的参数被冻结。V8 使用一个独立 geometry Value encoder，
因此 pose loss 可以训练 Value/evidence 映射而不会偷偷改变 correspondence。

pose head 使用 `evidence_only`：不再把 frozen L0 camera hidden 送入 residual MLP，避免模型用
帧特定 camera feature 绕过匹配。L0 只提供需要被修正的基础位姿。pose loss 仍为旋转矩阵
MSE 与 camera-center SmoothL1 的组合。

阶段 B 使用各分支共同的 inference-time active support，不用“是否存在 GT match”再次筛选
pose 训练帧。训练前后的 `p_ij` 必须 bit-exact，CSV 中对应
`matching_frozen_exact=1`；主候选与 geometry/SAM-off 控制还必须满足
`control_support_exact=1`。

无 mature instance 时 `active_frames=False`，refined pose 必须逐元素等于 frozen L0；frame 90
始终保持 gauge 不变。

## 4. V8.0 与 V8.1 分支

| CSV 分支 | Affinity | Value | 定位 |
|---|---|---|---|
| `geometry_match` | geometry | geometry | 无 SAM descriptor 控制 |
| `sam_match` | SAM | geometry | 只看 SAM affinity |
| `sam_geometry_match` | SAM + geometry | geometry | V8.0 主候选 |
| `sam_geometry_train_sam_off` | geometry（SAM 关闭） | geometry | 同结构 trained-off 控制 |
| `sam_geometry_no_match_supervision` | 未训练的 SAM + geometry | geometry | 同初始化 `λ_match=0` 控制 |
| `sam_geometry_dual_value` | SAM + geometry | geometry + SAM | V8.1 report-only 容量消融 |

V8.1 双 Value 的 point evidence 额外包含：

```text
[s_current, s_history_transported,
 s_current - s_history_transported,
 s_current * s_history_transported]
```

它不会参与 V8.0 是否成功的判定。只有在 V8.0 已通过，且双 Value 在 held-out pose 上继续
改善，才能认为传输 SAM Value 有额外价值。

## 5. CSV 指标与控制

匹配指标包括 `test_match_loss`、`test_match_coverage`、Top-1 accuracy、预测分布的期望 GT
3D 距离，以及 SAM/geometry projection 梯度范数。位姿指标包括 rotation、camera-center
error、pose loss 和相对 frozen L0 的提升。

因果控制包括 `sam_off`、`wrong_sam_identity`、`shuffle_sam_time`、纯 geometry、同结构
trained-SAM-off、同初始化但不训练 matcher 的 `λ_match=0` 分支，以及
reference/inactive bit-exact fallback。主候选还必须在 held-out matching 和 pose 上同时超过
这个 no-match-supervision 控制。

注意：geometry-only 分支仍使用 SAM tracking mask 和 persistent slot 来规定实例区域；它隔离的
是“局部 SAM descriptor 是否在相同区域和身份支持下提供额外对应信息”，不是完全移除 SAM
分割/追踪系统。

## 6. 时间泛化协议

所有分支使用相同 `90:15:525` 序列、相同动态实例 cache 和相同 frozen L0：

| Fold | matcher/pose 训练 | held-out 测试 |
|---|---|---|
| short | 270–345 | 360–405 |
| medium | 270–405 | 420–465 |
| long | 270–465 | 480–525 |

这是同场景时间外推，不是跨场景泛化。

## 7. 运行

```bash
zsh streaming_couping/commands_v80_supervised_correspondence.txt
```

命令默认在 cache 缺失时使用三张物理卡，训练阶段只使用 `V80_GPU` 指定的一张卡。已有 cache
会直接复用。常用覆盖：

```bash
V80_GPU=1 V80_MATCH_STEPS=1200 V80_POSE_STEPS=1200 \
zsh streaming_couping/commands_v80_supervised_correspondence.txt
```

主结果：

```text
outputs/streaming_couping_v80_supervised_correspondence/v80_supervised_correspondence.csv
```

重复运行会按签名续跑。设置 `V80_FRESH=1` 忽略 checkpoint 重新训练；设置
`V80_REBUILD_CACHE=1` 才重新运行 SAM3.1 和 StreamVGGT。

## 8. 结论边界

V8 仍不直接更新 StreamVGGT depth 或 dense pointmap。它验证的是 SAM3.1 descriptor 是否改善
跨视角局部几何对应，以及这种对应是否能改善相机位姿。点云改善仍只可能来自位姿对齐更稳定，
不能表述为 SAM 已直接优化单帧 pointmap。
