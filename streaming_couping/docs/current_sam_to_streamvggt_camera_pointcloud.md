# 当前 SAM3/SAM3.1 如何帮助 StreamVGGT 的相机与点云

> 代码状态：2026-07-31。本文只描述当前仓库仍可运行的路径，并严格区分“已经接入”
> 和“已经有实验结果”。

## 1. 一句话结论

SAM3/SAM3.1 不直接输出相机位姿，也不直接生成三维点。它为 StreamVGGT 提供三类原始
模型缺少的信息：

1. **跨帧实例对应**：哪些像素属于同一个物体；
2. **实例外观描述**：mask 内冻结 SAM backbone feature 的统计；
3. **空间作用范围**：哪些 camera/patch token、三维点和图像射线可以使用实例约束。

系统再把这些信息和 StreamVGGT 自身的 camera hidden、pointmap、depth、pose、intrinsics
组合起来：

- 对相机：用 persistent instance token 条件化 camera feature，学习有界旋转或 SE(3) 修正；
- 对点云：只在可信实例区域修改 StreamVGGT 的多层 DPT patch token，再由冻结的
  depth/point head 解码；
- 对相机中心：在可信实例像素内，用点云与图像射线进行解析式 angular-Huber 拟合；
- 对不可靠帧：保持或回退到 raw StreamVGGT。

因此更准确的表述不是“把 SAM token 直接塞进 StreamVGGT”，而是：

```text
SAM mask/feature
  → 物体级时序对应与实例 token
  → 条件化 StreamVGGT camera/patch feature
  → 受限的位姿和点云修正
```

## 2. 当前版本边界

仓库现在有两类相互隔离的证据：

| 路径 | SAM 版本 | 相机 | learned pointmap | ray solver | 当前用途 |
|---|---|---:|---:|---:|---|
| V4 coverage-first | 原始 SAM3 | 是 | 是 | 是 | 位姿收益和 mask 覆盖率研究主线 |
| V5 adaptive-best | 原始 SAM3 | 是 | 是 | 是 | 当前完整的 pointmap 安全对照 |
| V6 | SAM3.1 | 是 | 正式命令未训练/评估 | 否 | 几何辅助分割和 camera/instance 机制实验 |

这里有一个重要限制：

- V6 的 SAM3.1 最终 mask 已写入 `tracking_masks_*`、`trusted_tracking_masks_*`、
  instance geometry、appearance 和 ray-support 所需缓存字段；
- 但当前 `commands_v6_camera_overfit.txt` 只训练/评估 camera 模型，明确关闭 pointmap
  branch 和解析式 ray solver；
- 所以不能把 V4/V5 的点云提升数值直接写成 SAM3.1 的点云结果。要得到这一结论，仍需用
  SAM3.1 cache 单独训练并冻结 pointmap 分支，再在未参与开发的序列上评估。

## 3. 完整的双向数据流

当前不是无限循环优化，而是一次有明确先后顺序的双向耦合：

```text
RGB 序列 + 参考帧实例 mask
        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
冻结 StreamVGGT                     raw SAM3.1 tracking
camera/pose/pointmap/confidence            │
        │                                 │
        └─参考物体 3D 点投影───────────────┘
                          │
                          ▼
              box + positive-point prompt
                          │
                          ▼
          SAM3.1 prompted mask 与 raw mask 竞争
                          │
                          ▼
                   最终实例 mask
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
         SAM appearance pooling   mask 内 StreamVGGT 3D 统计
                 │                 │
                 └────────┬────────┘
                          ▼
             causal persistent instance token
                 ┌────────┴───────────┐
                 │                    │
                 ▼                    ▼
        camera feature fusion   DPT patch feature fusion
                 │                    │
                 ▼                    ▼
          refined rotation       refined pointmap/depth
                 └────────┬───────────┘
                          ▼
            trusted instance point-to-ray fit
                          │
                          ▼
                 refined camera center
```

为避免自增强错误，SAM3.1 几何提示使用的是**冻结 raw StreamVGGT** 的参考 pointmap、raw
pose 和 intrinsics，不使用 learned pointmap 或 refined camera 再反复提示 SAM3.1。也就是说，
当前实现是可分析的一次前向流程，不是可能发散的闭环迭代。

## 4. 第一步：StreamVGGT 反向帮助 SAM3.1 分割

这一部分只属于当前 V6。

### 4.1 参考物体三维化

系统从参考帧 mask 中选取 StreamVGGT 高置信度世界点：

```text
X_ref = {P_ref(u) | u ∈ M_ref, confidence(u) ≥ τ}
```

当前阈值来自 V6 配置：

```text
geometry_prompt_point_confidence_threshold = 0.30
```

点数过多时进行确定性采样。这里不使用 StreamVGGT hidden feature，只使用容易替换的标准
几何输出：pointmap、confidence、pose 和 intrinsics。

### 4.2 投影到目标帧

对目标帧 `t`，将 `X_ref` 通过 raw StreamVGGT 的 `W2C_t` 和 `K_t` 投影到图像：

```text
x_t ~ K_t · W2C_t · X_ref
```

投影后再与当前帧 raw pointmap 做深度/三维支持检查，去掉遮挡、越界和明显不一致的点，得到：

- `box_mask`：物体可能出现的粗范围；
- `positive_mask`：几何支持较强的稀疏正点。

这是一个轻接口。后续如果把 StreamVGGT 换成 HorizonStream 或其他几何模型，只要仍能提供
pointmap、confidence、pose 和 intrinsics，就可以继续生成相同的两张布尔 mask。

### 4.3 SAM3.1 候选与无 GT 选择

每帧同时保留：

1. raw SAM3.1 tracking mask；
2. 使用 text/box/positive points 得到的 prompted candidates。

候选分数为：

```text
geometry_score
  = 0.55 × positive-support recall
  + 0.30 × inside-box precision
  + 0.15 × SAM3.1 score
```

如果 raw mask 的正点覆盖率低于 `0.70`、box precision 低于 `0.10` 或 raw 为空，则认为 raw
不可靠。此时 prompted candidate 的 support 和总分都至少提高 `0.05` 才替换 raw；raw 已经
可靠时，两项都至少提高 `0.10`。另外要求候选/raw 面积比位于 `[0.5, 4.0]`。

任何条件不满足都逐帧保留 raw，因此几何提示是一个有回退的修正器，不是强制覆盖 SAM3.1。

当前部署策略名：

```text
v6_sam31_adaptive_positive_compete_010
```

已记录的 mask 选择结果：

| 开发范围 | mean IoU delta 相对 raw SAM3.1 | worse frames |
|---|---:|---:|
| train + validation 加权 | +0.04082 | 0 |
| test report-only | +0.02021 | 0 |

这些数值只说明当前选择实验，没有证明其他场景同样无退化。

## 5. 第二步：把 SAM mask 变成可用于几何的实例观测

最终 mask 本身还不能直接修正位姿。系统先为每帧、每实例构造三组特征。

### 5.1 SAM appearance

当前 V6 使用冻结 SAM3.1 `detector_fpn2` 特征。对实例 `k` 的 mask 做通道均值和标准差池化：

```text
a_tk = [mean(F_t(u)), std(F_t(u))],  u ∈ M_tk
```

这不是 SAM3.1 decoder 的现成 object token，而是本系统显式构造的 mask-pooled appearance。
长序列中 appearance backbone 按小批次运行，最终特征保存到 CPU cache；这只改变显存占用，
不改变方法定义。

### 5.2 物体三维描述

在 mask 内采样 StreamVGGT world points，构造 20 维几何描述，包含：

- 物体中心；
- covariance eigenvalues；
- 轴向 extent；
- 相对 persistent object map 的 registration translation；
- registration fitness 和 RMSE ratio；
- shape similarity；
- point confidence、mask area 和 point count；
- 多实例 translation consensus residual；
- frame gap。

质量向量为：

```text
q_tk = [track_confidence, geometry_confidence, static_score]
```

### 5.3 相机投影残差描述

camera 分支还可以使用 27 维 pose/reprojection geometry，例如：

- observed/projected mask centroid 及其 residual；
- observed/projected 2D covariance 及其 residual；
- observed/projected area；
- projection coverage 和 IoU；
- mask 内 ray mean/variance；
- persistent map point count 和 object spread。

这类特征表达的是“当前观察到的物体位置/形状”和“按当前相机与历史物体几何投影后应看到的
位置/形状”之间的偏差，因此比单纯外观更接近相机误差的可观测量。

## 6. 第三步：身份状态和因果实例 memory

系统不能把每个 SAM mask 都当作同一物体，否则相似外观错检会直接污染 camera 和 pointmap。
所以 mask 内三维点会与参考帧/历史 object map 做有界 registration：

| 状态 | 含义 | memory 写入 | camera 使用 | pointmap/ray 使用 |
|---|---|---:|---:|---:|
| `MATCH` | 三维身份支持充分 | 是 | 是 | 是 |
| `UNKNOWN` | 点太少、遮挡或 registration 证据不足 | 否 | 按版本低权重或不用 | 否 |
| `MISMATCH` | 点很多但几乎零几何支持，明确空间冲突 | 否 | V4/V5 拒绝；V6 可低权重软化当前帧 | 否 |
| `ABSENT` | 当前没有 mask | 否 | 有历史 memory 时可按版本读取 | 否 |

V6 的 `soft_unknown_strict_memory` 只把旧缓存中“零 correspondence 的 MISMATCH”在**当前帧**
软化成低可靠度 UNKNOWN，默认权重 `0.25`；它仍然不能写入 memory。这样做是为了减少正确
mask 因视角变化和参考点稀疏被连续硬拒绝的问题，同时防止长期污染。

### 6.1 persistent token

对每个实例维护 appearance memory 和 geometry memory。只有可靠 `MATCH` 可以更新：

```text
m_t = μ m_(t-1) + (1-μ) x_t
μ = 0.90
```

tokenizer 输入不是只有当前观测，而是：

```text
[current,
 memory,
 current - memory,
 quality,
 memory age]
```

经 MLP 编码为 persistent instance token。开始新 clip 时 memory 清空；按序处理帧，不读取
未来观测。

这一步是 SAM 能帮助相机的关键：SAM 提供同一物体的时序对应，memory 提供该物体历史上
可靠的外观与几何状态，而 `current-memory` 提供当前相机/点云与历史锚点不一致的信号。

## 7. SAM 如何帮助相机位姿

当前有两套相机实现，不能混为一个模型。

### 7.1 V4/V5：修改 StreamVGGT camera hidden，再复用冻结 CameraHead

取 StreamVGGT aggregator 最后一层的 camera hidden `C_t`，以 camera hidden 为 query、
persistent instance tokens 为 key/value 做 cross-attention：

```text
r_t = W_zero · Attention(C_t, I_t, I_t)
C'_t = C_t + sigmoid(g) · r_t
```

`W_zero` 从全零初始化，因此：

- 初始输出精确等于 raw StreamVGGT；
- `module_off` 精确等于 raw；
- 没有有效实例时 residual 为零；
- 训练只学习相对预训练模型的修正，不重新学习整套 pose 表示。

然后使用同一个冻结 CameraHead，按 StreamVGGT 原本的逐帧 causal KV 路径分别解码 raw
hidden 和 refined hidden。

#### V4 additive encoding

V4 计算冻结 CameraHead 输出差值，并加到缓存的 baseline pose encoding：

```text
pose_v4 = pose_raw + [Head(C') - Head(C)]
```

V4 camera token 主要使用 appearance；UNKNOWN 不进入 camera。优点是两段序列上位姿收益
较强，缺点是 pose encoding 的各分量仍是联合残差。

#### V5 bounded SO(3)

V5 从 raw/refined CameraHead 解码的旋转差中提取 `ΔR`，转为 axis-angle 后限制到最大 `5°`：

```text
ω = clamp(Log(R_decoded_refined R_decoded_raw^T), 5°)
R_v5 = Exp(ω) R_raw
```

V5 在 learned camera 步骤只写入旋转，baseline translation 和 intrinsics 保持不变。相机中心
交给后面的解析式 ray solver，避免 learned camera head 和几何 solver 同时自由修改平移。

### 7.2 V6：feature replacement + 新的 pose correction head

V6 不再把 attention 输出残差加回 camera hidden。它先得到 camera feature `C` 和 attended
instance feature `I`，再构造：

```text
Z = Merger([C, I, C ⊙ I, C - I])
```

pose correction head 只读取 `Z`。因此 feature fusion 是 replacement，而最终位姿仍锚定 raw：

```text
Δξ = Head(Z)
T_v6 = Exp(Δξ) · T_raw
```

最后一层仍全零初始化；参考帧强制不更新。V6 还保留：

- `camera_only`：只验证 camera hidden 自身可学习性；
- `instance_only`：只验证 persistent instances 可学习性；
- `fusion`：验证二者联合；
- 3DoF rotation、world-center、camera-frame translation 等职责解耦 head。

专门化组合重建时满足：

```text
t = -R C
W2C = [R | t]
```

而不是直接拼两个 W2C translation column。最终组合只有 rotation 和 instance-center 两个分支
都 active 时才应用；没有 persistent instance 支持则整帧回退 raw，避免 camera-only rotation
在无实例帧单独泄漏到结果。

### 7.3 为什么实例理论上能帮助 camera

对于静态物体，世界中的物体中心、尺度和形状不应随相机运动改变。若 camera pose 有误，会
同时表现为：

- 参考三维物体投影到错误图像位置；
- observed/projected centroid 和 covariance 不一致；
- 同一实例跨帧 point cloud 无法重合；
- point-to-ray residual 变大；
- 多个静态实例给出的平移提案不一致。

SAM 的价值是建立“这是同一个物体”的对应关系，使这些残差能够在物体级聚合，而不是在整幅
图像中被动态区域、重复纹理或背景像素淹没。

但一个网络能在五帧上过拟合，只证明容量足够。要证明实例真的有用，需要同时满足：

- `instance_off` 后退化；
- time shuffle/wrong geometry 后退化；
- 固定 checkpoint 在未参与训练/调参的 clip 或 scene 上优于 raw；
- 收益与 active persistent instances 相关；
- 无实例时严格回退 raw。

## 8. SAM 如何帮助点云

完整 learned pointmap 路径当前由 V4/V5 实现。

### 8.1 多层 patch token 条件化

StreamVGGT 的冻结 depth/point DPT head读取 aggregator 第：

```text
4, 11, 17, 23
```

层 token。系统保持 camera/register prefix 不变，只对 patch tokens 做实例 cross-attention：

```text
Z'_l = Z_l + sigmoid(g_l) · W_zero,l · Attention(Z_l, I, I)
l ∈ {4, 11, 17, 23}
```

四层分别学习，不共享写入投影；然后把四层 refined tokens 交回**冻结**的 StreamVGGT
depth head 和 point head：

```text
D_refined, conf_D = FrozenDepthHead(Z'_4,Z'_11,Z'_17,Z'_23)
P_refined, conf_P = FrozenPointHead(Z'_4,Z'_11,Z'_17,Z'_23)
```

SAM 没有直接预测 XYZ。它通过 persistent instance token 改变 DPT 原本消费的多尺度
几何 feature，让冻结 head 在实例区域输出不同的 depth/world point。

### 8.2 mask-local 空间写入

在 strict identity 模式下，`geometry_trusted_masks` 下采样到 patch grid，并可膨胀一个 patch。
只有 mask 覆盖的 patch 可以保留 learned update：

```text
Z_out(u) =
  Z_refined(u), u ∈ trusted instance patches
  Z_raw(u),     otherwise
```

DPT 解码后再做一次像素级 fallback：

```text
P_out(u) =
  P_refined(u), u ∈ geometry_trusted_mask
  P_raw(u),     otherwise
```

depth、depth confidence、world point 和 world confidence 都使用相同规则。这避免一个错误
instance token 通过全局 DPT feature 改坏背景，也是 `module_off_exact` 能成立的重要原因。

### 8.3 训练约束

V4/V5 的 pointmap 分支训练同时使用：

- aligned pointmap loss；
- scale-invariant depth loss；
- fixed-reference depth loss；
- 实例跨帧 trimmed rigid/centroid loss；
- camera encoding、relative rotation、translation direction loss；
- residual regularization。

SAM mask 在这里既决定实例刚性损失的采样区域，也决定 token update 的空间范围。

### 8.4 V5 的无 GT pointmap 安全回退

V5 只在两个预定义候选之间选择：

```text
P(0) = raw StreamVGGT pointmap
P(1) = learned pointmap
```

对 learned pointmap 运行相同的实例 point-to-ray solver，并统计非参考帧接受率：

```text
support_ratio = fit_accepted / nonreference_frames

support_ratio >= 0.75 → 使用 learned pointmap
support_ratio <  0.75 → 使用 raw pointmap
```

这个 gate 不读取 GT、ATE 或 pointmap GT error。它只回答：当前 clip 的实例点和图像射线是否
足够一致，能够支撑 learned geometry。

### 8.5 直接 pointmap 改善与 pose-induced PLY 改善要分开

评估时必须区分：

1. **direct pointmap**：冻结/learned point head 直接输出的世界点是否改变；
2. **reposed point cloud**：同一 depth/point 经 refined camera pose 重建后是否改变；
3. **object PLY selection**：使用 SAM mask 从 full scene 中截取哪些点。

V6 camera-only 即使让轨迹更好，也没有修改 DPT pointmap；如果某个重建 PLY 因 pose 更好而
更整齐，那是 pose-induced improvement，不能写成 learned pointmap improvement。

同样，`ours_predicted_object.ply` 与 GT 的差异同时包含 mask 和几何误差；
`ours_on_gt_mask.ply` 才是在固定 GT mask 下单独比较 pointmap 几何。

## 9. 解析式 point-to-ray 如何把点云收益反馈给相机

对每个可信实例像素，已有：

- learned world point `P_i`；
- pixel 与 intrinsics 形成的相机射线；
- refined rotation 把射线转到世界方向 `d_i`；
- point/instance confidence 权重 `w_i`。

相机中心 `C` 应使 `P_i-C` 与 `d_i` 共线。系统用 angular-Huber robust objective 求解：

```text
C* = argmin_C Σ_i w_i · Huber(angle(P_i - C, d_i))
```

求解后检查：

- 有效/保留点数；
- condition number；
- point-ray 和 angular residual；
- proposed center shift；
- 最大 residual 和最大 center shift。

通过才应用：

```text
C_out = C_fallback + blend · (C* - C_fallback)
```

否则保持 learned/raw fallback center。W2C translation 最后按：

```text
t_out = -R_out C_out
```

重建。参考帧是否固定、blend 和接受阈值由版本配置决定。

这一分支解释了为什么 learned pointmap 和 camera rotation 可以互补：rotation 决定射线方向，
pointmap 提供射线应穿过的世界点，solver 再恢复相机中心。

## 10. 当前已有的相机和点云证据

### 10.1 V4/V5 完整系统

| clip | Raw ATE | V4 ATE | V5 ATE | Raw pointmap | V4 pointmap | V5 pointmap |
|---|---:|---:|---:|---:|---:|---:|
| 90–240 | 0.36879 | 0.20464 | **0.15932** | 0.15258 | **0.14037** | 0.14045 |
| 492–589 | 0.36079 | **0.33316** | 0.35361 | **0.22446** | 0.22644 | **0.22446** |

解释：

- 第一段 SAM instance guidance 同时帮助 pose 和 pointmap；
- 第二段 V4 pose 改善更大，但 learned pointmap 轻微退化；
- V5 检测到 ray support 只有 `2/5`，因此保持 raw pointmap，避免进一步退化；
- 这些 pointmap 结果来自原始 SAM3，不是 SAM3.1。

### 10.2 V6 camera

历史 V6 camera 实验说明 camera-only、instance-only、fusion 都有拟合能力，并在同场景
held-out/cross-clip 上出现过改善；但这些旧数值来自原始 SAM3 cache。切换 SAM3.1 后应重新
训练和报告，不能把历史表直接标成 SAM3.1 camera 结果。

当前新增长序列压力测试为：

```text
30 frames: 90, 105, 120, ..., 510, 525
```

它复用 frame 90 参考视角，用于验证完整因果历史、显存和长序列稳定性，不是独立泛化证明。

## 11. GT 使用边界

| 阶段 | GT 使用情况 |
|---|---|
| 当前参考实例初始化 | 主实验使用 ScanNet++ GT instance ID 和参考帧 mask |
| StreamVGGT→SAM3.1 几何提示 | 不使用目标帧 GT mask/pose |
| raw/prompted mask 竞争 | 不使用 GT；只看几何 support、box 和 SAM score |
| identity/memory | 不使用 GT pose；使用预测 mask 和 frozen StreamVGGT geometry |
| V4/V5 训练 | 使用 GT pose、depth、pointmap supervision |
| V6 camera 训练 | 使用训练帧 GT pose |
| 推理选择和 ray gate | 不读取测试 GT 指标 |
| 评估/GT-world 对齐/GT object PLY | 使用 GT，仅用于指标和可视化 |

所以当前系统应表述为：

> 给定参考实例 mask 的 SAM3/SAM3.1 + StreamVGGT 时序实例引导优化。

还不能表述为完全无 GT 初始化的开放词汇部署系统。

## 12. 失败模式和当前保护

| 失败 | 影响 | 当前保护 | 未解决部分 |
|---|---|---|---|
| SAM 漏检 | 没有当前实例支持 | 可读取可信历史 memory；无实例回退 raw | 长时间漏检后 token 过时 |
| 相似外观错检 | 错误 token 污染 pose/pointmap | 3D identity、MATCH-only memory、mask-local update | raw geometry 自身错时可能误判 |
| 大视角变化/遮挡 | registration 零对应 | UNKNOWN；V6 低权重软化当前帧 | 软化过多可能增加错检 |
| 动态物体 | 破坏静态锚点假设 | static score、多实例 consensus | 没有显式动态模型 |
| pointmap 质量差 | geometry prompt 和 ray fit 同时受损 | confidence、support gate、raw fallback | 两个模块误差相关，不是独立证据 |
| 没有 active instance | learned correction 无依据 | reference/inactive frame bit-exact raw | camera-only 可改善的信息不会用于最终实例候选 |
| learned pointmap 局部退化 | 物体区域 XYZ 变差 | V5 support-ratio 选择 raw | gate 只覆盖 ray-support 类型失败 |

## 13. 代码位置

| 功能 | 文件 |
|---|---|
| StreamVGGT 几何提示 | `src/streamvggt_geometry_prompt.py` |
| SAM3.1 raw/prompted mask 竞争 | `src/v6_geometry_segmentation.py` |
| V6 SAM3.1 cache 接入 | `src/learned_pose/cache.py` |
| appearance/geometry/pose residual | `src/learned_pose/observations.py` |
| V4/V5 persistent token 与 camera/patch fusion | `src/learned_pose/model.py` |
| V4/V5 camera 与 frozen DPT 解码 | `src/learned_pose/pipeline.py` |
| point-to-ray camera center | `src/learned_pose/ray_pose.py` |
| V6 feature merger 与 SE(3)/3DoF head | `src/learned_pose/v6_camera_fusion.py` |
| V6 camera 实验 | `scripts/run_v6_camera_overfit.py` |
| 30 帧多卡 StreamVGGT | `src/backbones/streamvggt_parallel.py` |

## 14. 汇报时可以压缩成五句话

1. SAM3.1 提供跨帧同一物体的 mask 和外观，StreamVGGT 提供相机、射线和三维点。
2. 系统只让三维身份可信的静态实例写入因果 memory，形成 persistent instance token。
3. instance token 一路条件化 camera feature，另一路只在实例区域条件化 DPT patch feature。
4. learned rotation 和 learned pointmap 再通过可解释的 point-to-ray solver恢复 camera center；
   不可靠帧或 clip 回退 raw StreamVGGT。
5. 当前完整点云收益由 V4/V5 原始 SAM3 验证；V6 已切换 SAM3.1 并验证分割接口，SAM3.1
   pointmap 的独立训练与跨场景验证仍是下一步。
