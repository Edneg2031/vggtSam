# SAM3.1 × StreamVGGT 本周实验总结

> 时间范围：2026-08-03 至 2026-08-06
>
> 研究目标：判断 SAM3.1 的实例分割、persistent identity 和 local descriptor 中，究竟哪一部分
> 能对 StreamVGGT 相机位姿估计提供可归因、可因果验证的帮助。
>
> 本周覆盖：V7.4、V7.5、V8 matcher-first、V8 O1–O2.7、V9 Stage O/O-R1/A0/A0-H/A1。

## 1. 本周结论摘要

本周最重要的结果不是“又训练了一个更大的融合模型”，而是把原本混合在一起的几个问题逐层
拆开，并分别给出了证实或证伪结果：

1. **动态实例流程已经成立。** 物体不需要出现在 frame 90；SAM3.1 可以在后续帧发现新实例，
   分配永久 slot，并从第二次因果观测开始提供历史信息。
2. **直接 pose residual 的过拟合不构成 SAM-token 证据。** uniform、geometry、SAM、错误输入
   等分支都能把长序列训练 loss 压到接近零。
3. **SAM mask/identity 在局部 fold 有价值，但没有稳定的 all-fold 位姿收益。** mask 选择
   StreamVGGT points、ID 建立 object history 后，显式 ICP/ray solver 只在 short/frozen 成功，
   medium/long 和 online memory 不稳定。
4. **三维显式路线的主要瓶颈已定位为 StreamVGGT predicted depth/K。** GT camera geometry 和
   GT-depth+GT-K 能稳定工作；predicted depth 即使配 GT K 也失败，predicted K 也会独立伤害
   位姿。全局 scale、affine 和 Sim(3) 均不能稳定修复。
5. **正确的实例内二维 correspondence 可以显著修正相对位姿。** 正确 pairs 进入固定极几何
   solver 后，short、medium、long 三折全部通过，rotation 与 translation direction 同时改善，且
   没有恶化帧。
6. **当前 SAM3.1 `detector_fpn2` local token 没有学出未来帧点级 correspondence。** V9 A1
   虽然训练 loss 下降约 85%–92%，但 held-out PCK 只有 0%–2.59%，所有 12 个测试帧的 pose
   都恶化；正常 SAM 也没有稳定超过 patch、uniform、trained-off 或输入扰动。

因此，截至本周结束，可以形成如下准确表述：

> 已证实：SAM 实例区域内的正确二维跨帧对应，能够通过固定极几何求解器改善 StreamVGGT 的
> 相对 rotation 和 translation direction。
>
> 尚未证实：这些正确对应能够由当前缓存的 SAM3.1 `detector_fpn2` local descriptor 产生。
>
> 已在当前协议下证伪：`detector_fpn2 + farthest-UV local32 + Linear Q/K matcher` 能在严格未来
> 帧提供稳定的 SAM-token 位姿收益。

## 2. 公共数据与实验设置

### 2.1 数据集与场景

- 数据集：ScanNet++ pinhole 2D/3D 处理结果；
- scene：`00a231a370`；
- 长序列：`00a231a370_90_525_step15_37_68_54`；
- 输入帧：

```text
90, 105, 120, 135, 150, 165, 180, 195, 210, 225,
240, 255, 270, 285, 300, 315, 330, 345, 360, 375,
390, 405, 420, 435, 450, 465, 480, 495, 510, 525
```

共 30 帧，每隔 15 帧采样一次。当前所有结果都属于**同一场景内的时间外推**，不能称为
跨场景泛化。

### 2.2 frame 90 的角色

frame 90 只定义相机坐标 gauge，不再作为固定物体 reference。物体可以在任意后续帧出生。
reference/gauge 帧在所有要求 exact 的实验中保持不变。

### 2.3 SAM3.1 动态实例

- instance source：`sam31_online`；
- 推理方式：forward-only discovery/tracking；
- prompts：`bed`、`wardrobe`；
- 最大 logical slots：8；
- logical slot 是运行时容量，不是 ScanNet GT instance ID；
- 首次发现永久分配 slot；
- birth 帧只能写入 memory，不能使用尚不存在的历史；
- 从第二次可靠观测开始才能形成 pose evidence；
- 当前帧预测完成后才允许写回 memory，严格避免未来泄漏。

实际发现了 3 条主要 track：slot 0/1 在 frame 90 出现，slot 2 在 frame 150 才出生。因此“参考
帧没有物体、物体中途出现”的工程流程已经真实运行过，不是待实现设想。

### 2.4 SAM local descriptor

- feature source：SAM3.1 image detector 的 `detector_fpn2`；
- SAM 输入分辨率：1008×1008 stretch resize；
- 统一缓存 grid：72×72；
- 每个实例/帧最多 32 个 local token；
- 采样：从 mask centroid 开始的确定性 farthest-UV sampling；
- descriptor 存储：FP16；
- local UV：归一化到 `[-1,1]`；
- SAM3.1 backbone 在训练 matcher/pose adapter 时冻结。

需要特别注意：这里的 local token 是逐图像提取的 detector backbone feature，不是 SAM3.1 video
tracker 的时序 memory feature。SAM persistent identity 来自 tracking 系统，但 local descriptor
本身没有显式时序记忆。

### 2.5 StreamVGGT 与缓存

- StreamVGGT 使用 full-history streaming 输入；
- backbone 冻结；
- 30 帧 observation cache 同时保存：
  - frozen camera/L0；
  - predicted depth、pointmap、intrinsics 和 latent；
  - SAM dynamic masks/slots；
  - SAM local32 feature/UV/valid；
  - GT world point、depth、W2C 和标定 K，仅供监督、oracle 与评分。

V7.4 建立本周公共动态实例 cache。之后 V7.5/V8/V9 尽可能复用同一份冻结观测，避免不同
backbone forward 造成额外变量。

### 2.6 主要时间 fold

V7.4、V8 matcher-first 和 V9 使用相同的 train-prefix → strict-future 协议：

| fold | 训练 current frames | 严格未来测试 |
|---|---|---|
| short | 270, 285, 300, 315, 330, 345 | 360, 375, 390, 405 |
| medium | 270–405，step 15 | 420, 435, 450, 465 |
| long | 270–465，step 15 | 480, 495, 510, 525 |

V7.4 的 camera-only L0 使用 frames 105–255 建立，后续各 residual fold 从同一 frozen L0 开始。
未来 target、descriptor 和 memory 不进入对应 fold 的训练前缀。

V7.5 由于验证的是 object-map history，而不是 matcher 训练，使用独立的 prefix/test 划分：

| fold | frozen history prefix | test |
|---|---|---|
| short | 90–255 | 270–345 |
| medium | 90–345 | 360–405 |
| long | 90–405 | 420–525 |

每个 V7.5 fold 同时测试 frozen-prefix 和 prediction-after-update 的 online causal memory。

### 2.7 证据与指标边界

不同阶段的主要指标不同：

- V7.4/V8 learned pose：冻结 L0 上的 pose loss/gain、active frame、输入扰动 damage；
- V7.5：mean camera-center error，rotation 保持原 StreamVGGT；
- V8 O1–O2.7：显式 Kabsch/Umeyama 的 rotation、center、fit、support 与 fallback；
- V9：relative rotation error、relative translation-direction angular error；
- V9 matcher：PCK@8px、mean EPE、dustbin accuracy、Sampson RMSE。

V9 的主要结论不涉及 metric translation scale。essential matrix 只能恢复 translation direction；
absolute center/trajectory 只作次要诊断，不参与 SAM-token 主命题。

## 3. SAM local descriptor 作为匹配权重，再由 adapter 输出 SE(3)

> 对应代码阶段：V7.3/V7.4。版本号只用于定位结果，下面按真实信息流说明。

### 3.1 SAM 用了什么，如何参与 StreamVGGT

这条路线使用的不是一个抽象的“fusion 分支”，而是下面这条具体路径：

```text
SAM3.1 tracking mask + persistent ID
  → 确定当前帧与历史帧属于同一实例的区域
  → 在 mask 内取 detector_fpn2 local descriptors，并采样为 local32
  → 将 SAM descriptor 插值到 StreamVGGT 的实例几何采样点
  → SAM current/history descriptor 经 Linear Q/K 产生相似度 logits
  → softmax 得到当前点到历史点的 transport probability
  → 用该 probability 搬运历史 StreamVGGT geometry Value
  → [当前 geometry、搬运后 geometry、差、乘积] 池化成实例 evidence
  → 与 frozen camera-L0 hidden 一起输入 learned SE(3) adapter
  → adapter 输出 rotation/translation residual，左乘到 StreamVGGT L0 pose
```

这里必须区分三种信息：

| 信息 | 来源 | 在流程中的作用 | 是否直接作为 Value |
|---|---|---|---:|
| mask/track ID | SAM3.1 video tracker | 限定实例区域、关联历史实例、控制 memory | 否 |
| local descriptor | `detector_fpn2` local32 | 产生 current-history matching 权重 | 默认否 |
| local geometry | frozen StreamVGGT | 被 matching 权重搬运，形成 pose evidence | 是 |

因此默认实现里，SAM token 只决定“从哪一个历史几何点取信息”，真正被搬运的是 StreamVGGT
geometry。最终 pose 仍由可训练 adapter 输出，所以即使 pose loss 降到零，也可能只是 adapter 根据
camera hidden、geometry 或固定帧规律拟合出来，并不能自动证明 SAM token 有效。

### 3.2 对照如何切断不同信息

| 代码名 | 实际含义 | 用来排除什么解释 |
|---|---|---|
| `uniform_transport` | 不看 SAM/geometry 相似度，平均搬运有效历史 geometry | 仅靠实例 support 和 adapter 是否已能拟合 |
| `geometry_transport` | 只用 StreamVGGT geometry Q/K 计算权重 | 收益是否来自 geometry，而不是 SAM descriptor |
| `sam_transport` | 只用 SAM local descriptor Q/K 算权重，但仍搬运 geometry | SAM descriptor 是否提供正确匹配 |
| `sam_geometry_transport` | SAM logits 与 geometry logits 相加后搬运 geometry | 两种 affinity 联合是否更稳定 |
| `trained-SAM-off` | 参数量和 adapter 相同，但训练时关闭 SAM | 增益是否只是额外参数/adapter 容量 |
| wrong-ID / shuffle-time | 保持形状和 support，破坏正确身份或时间 | 输出是否因果依赖正确 SAM 内容 |

这一阶段不再只问“能不能拟合”，而是问：在严格未来帧中，正确 SAM descriptor 是否同时超过
geometry-only、trained-SAM-off，并且在 wrong-ID/shuffle-time 后稳定变差。

### 3.3 结果

| fold | SAM+geometry vs L0 | vs geometry | vs trained-SAM-off | pass |
|---|---:|---:|---:|---:|
| short | +22.81% | -5.58% | -5.86% | 0 |
| medium | +10.26% | -2.52% | +8.29% | 0 |
| long | +22.01% | +9.21% | +3.29% | 0 |

long fold 虽然超过两个控制，但 `sam_off / wrong-ID / shuffle-time` 的 damage 约为
`-0.88% / -0.88% / -0.67%`，即破坏正确 SAM 输入反而略好。三个 fold 和 all-fold SAM causal
pass 均为 0。

### 3.4 归因结论

**证实：**

- 动态 birth、永久 slot、严格因果 memory 和无成熟实例 exact L0 fallback 已实现；
- transport + SE(3) adapter 具有训练容量；
- 同一序列训练帧 loss 可被压低，但这是 **adapter capacity** 证据。

**证伪或未通过：**

- 未证实 SAM descriptor 在未来帧提供超出 geometry 的稳定收益；
- 输入破坏没有形成稳定伤害，因此不能把 pose gain 归因于正确 SAM 内容；
- “训练到零”只能说明 adapter 会拟合，不能作为 SAM token 学到物理 correspondence 的证据。

## 4. SAM mask/ID 选择 StreamVGGT 点云，再显式求相机中心

> 对应代码阶段：V7.5。这里完全不用 SAM appearance/local token，也没有 learned pose adapter。

### 4.1 目的与方法

SAM 在这里提供两样东西：mask 决定从 StreamVGGT pointmap 中取哪些点，persistent ID 决定这些点
写入哪一个历史 object map。位姿修正由固定几何算法完成：

```text
SAM mask 内 frozen StreamVGGT points
  → bounded translation ICP 到历史 object map
  → V5 angular-Huber point-to-ray camera-center solver
  → 数学/残差/shift gate
  → refined center 或 exact raw fallback
```

rotation 始终保留 raw StreamVGGT。

### 4.2 对照

- full image；
- SAM bbox；
- deterministic same-area random region；
- stale SAM mask；
- current-only；
- correct persistent history；
- wrong identity；
- GT mask oracle（仅上界）。

### 4.3 结果

- frozen-prefix all-fold region pass：0；
- online-memory all-fold region pass：0；
- frozen-prefix all-fold history pass：0；
- online-memory all-fold history pass：0；
- short/frozen 的 region 与 history 单独通过；
- medium/long 不稳定；
- long/online 相对 raw 下降约 3.51%。

### 4.4 本阶段证据

**证实：** SAM mask/identity 在局部时间范围内确实能提供比若干区域控制更好的相机中心约束。

**证伪或未通过：** 正确 SAM region 和 persistent history 没有在所有 fold、两种 memory mode 下
稳定改善相机中心。V7.5 也不能被引用为 SAM descriptor/token 证据。

## 5. 显式监督 matching，再让 evidence-only adapter 输出 pose

> 对应代码阶段：V8 matcher-first。目标是阻断 camera hidden 直接记忆 pose 的捷径。

### 5.1 目的

上一条 learned route 的问题是 matcher 和 pose head 被同一个末端 pose loss 驱动，adapter 可以绕过
correspondence。这里将其拆成：

1. Stage A：用 GT-world pseudo-match 单独监督 `p_ij`；
2. 冻结 matcher；
3. Stage B：evidence-only pose，不读取 camera hidden；
4. geometry、trained-SAM-off、no-match-supervision、dual Value 等控制共享相同 support。

完整信息流为：

```text
SAM local descriptor + StreamVGGT local geometry
  → 预测 current-history correspondence probability p_ij
  → GT-world pseudo-match 单独监督 p_ij
  → 冻结 matcher
  → p_ij 同时搬运 history geometry；dual-Value 时也搬运 history SAM descriptor
  → current/transported/difference/product 组成 evidence
  → evidence-only adapter 输出 SE(3) residual
```

与第 3 节相比，adapter 不再读取 camera hidden；但它仍是可训练 pose 输出模块，所以只有 matcher
本身的 held-out matching 指标、SAM 输入破坏和 parameter-matched controls 同时通过，才能归因给
SAM token。

### 5.2 结果

| fold | SAM+geometry vs L0 | vs geometry | fold causal pass |
|---|---:|---:|---:|
| short | +58.83% | +4.99% | 0 |
| medium | +12.79% | -1.48% | 0 |
| long | -37.78% | -4.76% | 0 |

其它诊断：

- held-out supervised query 约为 short/medium/long `10/2/11`，支持非常稀疏；
- short 的 no-match-supervision pose loss `0.0947`，优于 supervised SAM+geometry 的 `0.1389`；
- SAM perturbation 没有稳定伤害 matcher；
- dual SAM Value 只在 long 相对 geometry Value 改善，short/medium 分别下降约 61.54%/10.56%，
  long 仍差于 L0。

### 5.3 本阶段证据

**证实：** 可以将 matching supervision 与 pose 学习解耦，排除 camera hidden/direct pose head
捷径。

**证伪或未通过：** 双 Value、显式 matcher supervision 和 evidence-only pose 仍没有形成 all-fold
SAM causal pass；short 的提升甚至不需要 match supervision，因此不能归因于 learned SAM match。

## 6. SAM mask 限定三维实例区域，再用固定 Kabsch 求 pose

> 对应代码阶段：V8 O1–O2.7。这里不使用 SAM descriptor，也不训练 adapter。

SAM mask 在这里仅用于限定实例区域；persistent ID 关联当前与历史实例。实例内三维点来自
StreamVGGT predicted depth/pointmap，再由固定 Kabsch/Umeyama 求相对变换。实验固定 GT-world
pseudo-correspondence，逐层替换 geometry、depth、K 和 history pose，判断三维路线到底在哪里失效：

```text
SAM mask + persistent ID
  → 当前/历史帧实例区域
  → 从 StreamVGGT depth/K 反投影三维点
  → 固定 GT pseudo-correspondence（只做上界诊断）
  → weighted/trimmed Kabsch
  → relative pose correction
```

### 6.1 正确 GT 三维几何下验证 support 与 solver（O1/O2.5）

support sweep 包括：

- local token：32/64；
- history：1/2/4；
- mutual/one-way GT-world NN；
- radius：0.10/0.15 m；
- weighted/trimmed Kabsch。

结果：

- 30/30 坐标审计通过；
- 8 个配置在 all-fold support reliability 上通过；
- 自动选择最轻配置：`K=64 / history=2 / mutual / 0.10m`；
- O1 GT camera geometry + GT history 在 weighted/trimmed Kabsch 下通过；
- predicted depth 的 O2-GT-history 和 O2-L0-history 均失败。

**证实：** 跨帧支持、坐标约定和显式 Kabsch 存在可行上界。

**证伪：** 当前 StreamVGGT predicted camera geometry 不是只要增加 correspondence 数就能修复。

### 6.2 逐项替换 predicted depth 与 predicted K（O2.6）

固定 `K=64 / history=2 / mutual / 0.10m`，分解结果：

| 分支 | all-fold 结果 |
|---|---:|
| direct GT camera O1 | 通过 |
| GT depth + GT K | 通过 |
| GT depth + predicted K | 失败 |
| predicted depth + GT K | 失败 |
| predicted depth + predicted K | 失败 |
| oracle scale depth | 失败 |
| oracle affine depth | 失败 |
| uncalibrated predicted geometry Sim(3) | 失败 |

**证实：** depth sampling/backprojection 和 solver 在 GT-depth+GT-K 下正确；predicted K 与
predicted depth 都会独立损害位姿。

**证伪：** predicted depth 的主要问题不是单一全局 scale/offset，也不能仅靠 Sim(3) 替代 SE(3)
解决；局部 depth shape 与跨帧一致性是更深层瓶颈。

### 6.3 固定/平滑 K，并在 SAM mask 内拟合 depth affine（O2.7）

比较：

- current predicted K；
- frame-90 reference predicted K；
- causal-median predicted K；
- GT K；
- raw/global-affine/SAM-instance-affine predicted depth。

结果：

- 三种 deployable predicted-K 策略均未全折通过；
- global affine + GT K 未全折通过；
- instance affine + GT K 未全折通过；
- instance affine 的 fold gain 高于 global affine：
  - short：49.89% vs 34.90%；
  - medium：27.73% vs 18.99%；
  - long：90.05% vs 81.11%；
- medium frame 435 仍失败，instance affine + GT K 相对 L0 约下降 30.89%。

**证实：** SAM mask 定义的实例区域对局部 depth calibration 有 E2 级上界价值。

**证伪或未通过：** 固定/平滑 predicted K 或实例内 affine depth 仍不足以稳定修正全部未来帧；
该结果仍不能归因于 SAM descriptor。

## 7. SAM local descriptor 预测二维对应，再由固定极几何修正相对 pose

> 对应代码阶段：V9 Stage O/O-R1/A0/A0-H/A1。这里不训练 pose adapter。

### 7.1 为什么转向二维路线

上一节已经证明 predicted depth/K 是三维显式求解的瓶颈。这条路线不再使用 predicted depth、
pointmap Kabsch 或 learned SE(3) head，而是验证：

```text
SAM mask/ID 限定同一实例
  → SAM current/history local descriptor 预测 2D correspondence
  → calibrated essential/epipolar solver
  → relative rotation + translation direction
```

主实验使用 ScanNet++ calibrated K。它是传感器标定，在本场景中合理，但对于无标定互联网图片
仍属于 oracle deployment 条件。

### 7.2 用密集 GT 可见表面对应验证二维 solver（Stage O）

GT mesh/depth/pose 只用于构造同一可见 surface point 的 2D label。初版 solver 结果：

| fold | rotation gain | direction gain | worse frames | pass |
|---|---:|---:|---:|---:|
| short | +17.54% | +27.80% | 1 | 0 |
| medium | -338.38% | -623.51% | 3 | 0 |
| long | +18.82% | -230.40% | 2 | 0 |

所有 frame 均有大量 visible correspondence，因此失败不是 coverage，而是 eight-point
initialization 与无限 history-center fusion 的求解问题。

### 7.3 修正初始化与相对位姿评分，不改变对应（O-R1）

O-R1 只修求解器，不改数据和 correspondence：

- eight-point 与 frozen L0 relative pose 双初始化；
- 用同一 observed correspondence 的 robust Sampson objective 选候选；
- 固定次数局部 refinement；
- 不读取 GT pose error；
- relative edge 指标不再混入单目不可恢复的 metric center scale。

结果：

| fold | rotation gain | direction gain | worse frames | relative pass |
|---|---:|---:|---:|---:|
| short | +75.65% | +92.00% | 0 | 1 |
| medium | +32.90% | +86.99% | 0 | 1 |
| long | +78.37% | +58.33% | 0 | 1 |

**证实：** 可见 surface 的正确 2D correspondence 具有稳定 relative-pose 上界。

**仍未证实：** absolute center/trajectory；单目 essential edge 没有新的 metric scale。

### 7.4 固定 current local32 位置，给出连续 GT history 对应（A0 all-edges）

A0 不训练 matcher。它保留缓存的 32 个 current local query 位置，用 GT depth/pose 将每个可见
query 连续重投影到 history image，再进入固定 O-R1 solver。

原 `all_edges_mean` 结果：

| fold | rotation gain | direction gain | worse frames | pass |
|---|---:|---:|---:|---:|
| short | +71.87% | +85.45% | 0 | 1 |
| medium | +8.76% | +43.43% | 1 | 0 |
| long | +62.62% | +18.34% | 1 | 0 |

逐 edge 诊断发现：

- frame 450：history 345 的 condition 416，aggregate `9.29→1.02`；history 435 的 condition
  1722，aggregate `22.29→57.42`；
- frame 495：history 405 的 condition 32.7，aggregate `16.14→0.54`；history 465/480 的
  condition 3454/3067，aggregate 分别 `22.07→103.17`、`13.46→40.51`。

correspondence 数、Sampson RMSE 与 cheirality 不能稳定区分好坏 edge；design condition/rank 是
唯一明显且不读取 GT error 的 observability 信号。

### 7.5 用可观测性选择一条 history edge（A0-H）

A0-H 不是新模型，而是一个预注册的、无效果阈值的 history 选择：

1. 仍计算全部严格因果 history edge；
2. 只在数学求解成功的 edge 中选择 `design_condition` 最小的一条；
3. tie-break：更大 rank ratio，然后更小 history index；
4. 没有成功 edge 时 exact L0 fallback；
5. 不读取 GT pose error。

结果：

| fold | active | rotation | direction | aggregate | gain | worse | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 4/4 | `3.073→0.816°` | `10.664→0.463°` | `13.737→1.279°` | 90.69% | 0 | 1 |
| medium | 4/4 | `2.227→0.570°` | `11.575→1.137°` | `13.801→1.707°` | 87.63% | 0 | 1 |
| long | 4/4 | `5.054→0.820°` | `20.694→6.057°` | `25.748→6.877°` | 73.29% | 0 | 1 |

**A0-H 证实：** 当前实例区域内的 32 个 current query 位置，加上一条可观测性好的历史 edge，
在给定正确连续 2D correspondence 时足以改善 relative pose。

**A0-H 没有证实：**

- SAM descriptor 能找出 correspondence；
- history frame 的实际 32 个离散 key 足以逼近连续 GT 投影；
- edge 选择能跨场景泛化。A0-H 规则是在当前数据的 per-edge 诊断后提出，并在同一场景 fold 上
  验证，因此它是 solver/support gate，不是独立的跨数据集结论。

### 7.6 用 SAM local descriptor 学习实际二维匹配（A1）

A1 固定 SAM3.1、StreamVGGT 和 O-R1 solver，只训练一个小型 matcher：

- query/key 输入：每实例 32 个 local descriptor；
- parameter-free channel canonicalization 到 256 维；
- Linear Q/K projection：64 维；
- 参数量：32,769；
- temperature：0.07；
- soft target sigma/radius：6px/12px；
- PCK：固定 8px；
- cycle weight：0.10；
- optimizer：AdamW；
- steps：800；
- batch pairs：16；
- learning rate：3e-4；
- weight decay：1e-4；
- 不训练 pose，不把 pose loss 回传给 matcher。

控制：

- `sam_match/normal`；
- `stream_patch_match`：相同 UV、相同 Q/K 参数量；
- `mask_uv_uniform`；
- `sam_train_off`：相同 matcher，训练时 descriptor 置零；
- 同一 SAM checkpoint 的 `sam_off`、wrong identity、shuffle time、fixed channel permutation。

结果：

| fold | train loss | held-out PCK | EPE | aggregate pose | worse | pass |
|---|---:|---:|---:|---:|---:|---:|
| short | `5.153→0.429` | 2.59% | 142.37px | `17.886→128.781°` | 4/4 | 0 |
| medium | `4.452→0.548` | 0% | 284.45px | `19.761→111.955°` | 4/4 | 0 |
| long | `4.098→0.595` | 0.80% | 321.93px | `25.988→154.722°` | 4/4 | 0 |

其它关键信号：

- correct history key coverage 上限约为 short 80.3%、medium 44.4%、long 47.8%，但实际 PCK
  远低于这个上限；
- normal SAM 的 Sampson RMSE 约 0.12–0.21，oracle 通常约 1e-4–1e-3；
- normal SAM 的所有 12 个测试 frame 都恶化；
- channel permutation 的 EPE 在三折都低于 normal SAM，long PCK 还更高；
- StreamVGGT patch 在 short/long 的 PCK 高于 normal SAM；
- perturbation 没有形成稳定、方向一致的 matching 与 pose damage。

**A1 证实：** 当前 Q/K 确实能降低训练 objective，说明参数和梯度链路不是完全断开。

**A1 证伪或未通过：**

- 训练 loss 下降不能推出点级 correspondence 学成；
- 当前 `detector_fpn2 + local32 + Linear Q/K` 没有未来帧 matching 泛化；
- 正常 SAM 内容没有稳定优于 generic patch、uniform 或 parameter-matched control；
- 因果扰动没有稳定伤害，因此没有 SAM-token → correspondence → pose 的 E3 证据。

## 8. 已证实有效的环节

这里的“有效”指某个具体组件或物理中间量获得了直接证据，不等于整个 SAM-token pose 方法已经
成功。尤其要把 adapter capacity 与 SAM information value 分开。

| 已证实有效的东西 | SAM 提供了什么 | 如何参与 StreamVGGT/pose | 直接证据 | 不能扩大成什么结论 |
|---|---|---|---|---|
| 动态多实例管理 | prompt-conditioned mask、persistent track ID | 新实例分配永久 slot；当前帧结束后写 memory；第二次观测才可参与 | slot 2 在 frame 150 中途出生；因果 memory 与 exact fallback 正常 | 不代表 descriptor 有位姿价值 |
| learned adapter 的拟合能力 | 可以有 SAM/geometry 输入，也可以关闭或破坏 SAM | evidence/camera hidden → MLP/SE(3) head → L0 pose residual | uniform、geometry、SAM、wrong-input 都能把训练 loss 压到接近零 | **有效的是 adapter 容量，不是 SAM token** |
| SAM mask/ID 的区域信息 | 二值 mask 与跨帧 ID，不含 appearance token | mask 选择 StreamVGGT points；ID 建立历史 object map | 局部 fold 优于 full/bbox/random/stale；instance-region affine 上界优于 global affine | 还不是稳定 all-fold pose 方法，也不是 token 证据 |
| 显式三维 solver 上界 | SAM 只限定实例区域；correspondence/geometry 使用 GT oracle | GT 3D geometry → Kabsch/Umeyama → relative pose | GT camera geometry、GT-depth+GT-K 全折可工作 | 不代表 predicted StreamVGGT geometry 足够 |
| 正确实例内二维 correspondence | mask/ID 规定同一实例；对应本身由 GT oracle 给出 | 正确 2D pairs → fixed essential solver → rotation + translation direction | O-R1 与 A0-H 三折均通过；A0-H aggregate gain 为 90.69%/87.63%/73.29%，无恶化帧 | 不代表 SAM descriptor 能产生这些 pairs |
| history edge 可观测性选择 | 不使用新的 SAM 特征 | 在多个因果 history 中选择 design condition 最好的一条 | 避开 frame 450/495 的退化 edge，A0-H 全折通过 | 尚未证明跨 scene 泛化 |

### 8.1 最关键的正面结论

当前真正被证实的是：

```text
如果已经有正确的实例内二维跨帧对应
  → 不需要 learned pose adapter
  → 固定极几何求解器就能改善 StreamVGGT L0 的相对 rotation 和 translation direction
```

这证明“实例 correspondence 可以帮助 StreamVGGT pose”这条物理路径成立，但正确对应目前来自
GT oracle，而不是当前 SAM local descriptor。

### 8.2 adapter 过拟合应如何表述

正确表述：

> transport/evidence + SE(3) adapter 有足够容量拟合这一个序列的 pose residual。

错误表述：

> 因为带 SAM 输入的 adapter 把 loss 降到零，所以 SAM token 能修正位姿。

后者不成立，因为 uniform transport、geometry-only、SAM-off 甚至错误 SAM 输入也能拟合。只要
存在可训练 pose head，它就可能使用 camera hidden、geometry、frame regularity 或参数容量直接记忆
训练 pose。只有正确 SAM 在 held-out 上超过参数匹配控制，并且 wrong-ID/shuffle-time 明显伤害
结果，才能把收益归因给 SAM token；当前没有出现这种证据。

## 9. 已证伪或当前实现无效的路径

“证伪/无效”均限定于本周的场景、输入特征、模型和严格时间协议，不是否定所有可能的 SAM 架构。

| 当前无效的主张或路径 | 实际尝试的计算路径 | 为什么判定无效 |
|---|---|---|
| 训练 pose loss 接近零即可证明 SAM token 有用 | SAM/geometry evidence → learned SE(3) adapter | uniform、geometry-only、SAM-off 和错误输入同样能过拟合；只能证明 adapter capacity |
| SAM descriptor affinity 能稳定帮助未来 pose | SAM local Q/K → transport history geometry → adapter | 三个时间 fold 的 causal pass 都为 0；破坏 SAM 后没有稳定下降 |
| 把 SAM descriptor 也作为 Value 就能解决问题 | 同一 probability 同时搬运 geometry 与 SAM Value → evidence-only adapter | short/medium 明显下降，long 仍差于 L0；无 all-fold pass |
| 显式 matching supervision 自动带来 SAM 因果收益 | GT pseudo-match 监督 matcher → 冻结 → evidence-only pose | no-match-supervision 可更好；SAM perturbation 没有稳定伤害结果 |
| SAM mask/ID object map 能稳定修正所有相机中心 | mask 内 StreamVGGT points → ICP/history map → ray-centre solver | 只在 short/frozen 局部通过，medium/long 与 online memory 不稳定 |
| predicted StreamVGGT 3D geometry 足以做实例 Kabsch | SAM instance region → predicted depth/K → GT correspondence → Kabsch | 即使 correspondence 为 GT oracle 也失败；predicted depth 和 predicted K 都是独立瓶颈 |
| 用简单 calibration 可修复 predicted geometry | fixed/median K、global/instance affine depth、scale 或 Sim(3) | 所有方案都没有 all-fold pass；depth 问题不是单一 scale/offset |
| 将所有历史 edge 平均会更稳 | 每个 history edge 求 pose，再聚合 | frame 450/495 的坏 edge 会压倒好 edge |
| 当前 SAM local token 能产生未来 2D 对应 | `detector_fpn2 local32 → Linear Q/K → 2D pairs → fixed epipolar solver` | train loss 下降 85%–92%，但 held-out PCK 仅 0%–2.59%，12/12 pose frame 恶化 |
| pose 输出因果依赖正确 SAM descriptor | normal 与 wrong-ID/shuffle-time/channel-permutation 对照 | 扰动没有稳定造成 matching/pose damage；channel permutation 的三折 EPE 反而都低于 normal |

### 9.1 对 SAM 各类信息的最终分类

| SAM 信息 | 当前状态 | 准确结论 |
|---|---|---|
| text prompts | 工程有效 | 目前只配置 `bed/wardrobe`，用于发现候选实例，不直接修正 pose |
| tracking mask | 局部有效/E2 | 能定义比全图更有意义的实例区域，但尚无稳定 all-fold pose 收益 |
| persistent ID | 工程有效、位姿收益不稳定 | 能建立因果 object history；正确 ID 还没有稳定改善所有 fold |
| pooled/global appearance token | 未形成因果证据 | 与 camera/geometry 一起进入 adapter 时无法隔离其贡献 |
| `detector_fpn2` local32 descriptor | 当前协议下无效 | 当前 Linear Q/K 没有学出 held-out 2D correspondence |
| 正确 instance-local correspondence | 有效，但来自 oracle | 能显著修正 relative pose；这是后续 descriptor 必须达到的目标中间量 |
| SAM3.1 tracker/memory temporal feature | 未测试 | 不能用 detector-FPN 的失败代替对 temporal memory feature 的结论 |

## 10. 仍未回答的问题

本周结果没有回答、因此不能提前下结论的问题：

1. **实际 32 个 history key 的离散 support 上界。** A0-H 使用的是 current local32 query 到连续
   GT history UV，不是严格的 32×32 token oracle。需要 A0-Q：GT 只能从实际 valid history key
   中选择最近 token。
2. **A1 是训练 objective 走捷径，还是时间泛化失败。** 当前只报告 train loss，没有 train-frame
   PCK/EPE。需要在不重新训练 pose 的情况下补 train/test matching audit。
3. **SAM3.1 tracker/memory feature 是否比 detector FPN 更适合 temporal correspondence。** 当前
   descriptor 是逐图像 `detector_fpn2`，不能把其失败推广到所有 SAM3.1 temporal feature。
4. **跨 scene 泛化。** 当前全部正式 fold 都来自同一 scene。
5. **无 calibrated K 时的二维路线。** V9 使用 ScanNet++ GT calibration；predicted K 已在 V8
   显示不稳定。
6. **从 translation direction 恢复稳定 metric trajectory。** 单目 essential edge 没有新尺度，
   当前不能声明 absolute center/trajectory 改善。

## 11. 下一步最小验证顺序

为了避免继续兜圈子，下一步只应补 matching 层的缺失证据，不再扩大 pose head 或 sweep solver：

### 11.1 A0-Q：离散 key oracle

```text
current local32 query
  → GT continuous history projection
  → 只能从实际 32 个 valid history key 中选最近 key
  → dustbin if unsupported
  → fixed best-condition edge policy
  → fixed O-R1 solver
```

- A0-Q 失败：32-key sampling/量化本身就是上界瓶颈，停止当前 A1；
- A0-Q 通过：离散 support 足够，继续检查 descriptor/matcher。

### 11.2 Train/test matcher audit

同一 checkpoint 同时输出：

- raw descriptor cosine Top-1；
- train-frame 与 held-out PCK/EPE；
- top-1 与 soft-expectation 两种 UV；
- visible-supported CE 与 dustbin CE；
- entropy、real-key mass、max probability；
- SAM、StreamVGGT patch、uniform 的同 support 对比。

判断：

- train PCK 也低：loss/dustbin/soft target 或 descriptor separability 有问题；
- train PCK 高、test PCK 低：时间过拟合，`detector_fpn2` 缺乏未来泛化；
- raw/trained detector feature 都失败：停止 detector-FPN 路线，之后如继续，只允许测试真正不同的
  tracker/memory feature，而不是再增大 Q/K 或 pose head。

## 12. 本周最终研究结论

本周已经把“为什么 SAM 目前没有稳定帮助 StreamVGGT pose”从一个模糊的 fusion 问题，缩小到
一个明确的 correspondence 问题：

```text
动态 SAM mask/identity          已实现
learned SE(3) adapter 拟合容量  已证实，但与 SAM token 贡献无关
正确实例区域具有几何信息        已证实
GT 3D/2D 显式 solver 上界       已证实
predicted 3D depth/K 路线        当前失败，瓶颈已定位
正确 2D correspondence 路线     已证实能改善 relative pose
detector_fpn2 local token matcher 当前无法产生未来正确对应
```

因此本周最稳妥的成果表述是：

> SAM3.1 的动态实例系统为 StreamVGGT 提供了有效的区域、身份和潜在几何支持；在 oracle 正确
> 对应下，这些实例区域能够显著改善相对相机位姿。但截至当前，尚无证据表明缓存的 SAM3.1
> `detector_fpn2` local descriptor 能在未来帧产生足够准确的点级 correspondence；当前 Linear
> Q/K 实现反而会通过错误对应显著破坏位姿。

详细历史与不可重复实验索引见
[SAM3.1 × StreamVGGT 实验账本](sam31_streamvggt_experiment_ledger.md)，V9 设计与求解器边界见
[V9 二维极几何位姿因果实验](v90_epipolar_token_causality.md)。
