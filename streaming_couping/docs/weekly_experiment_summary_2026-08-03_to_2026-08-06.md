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
3. **SAM mask/identity 在局部 fold 有价值，但没有稳定的 all-fold 位姿收益。** V7.5 的显式
   region/history solver 在 short/frozen 成功，但 medium/long 和 online memory 不稳定。
4. **三维显式路线的主要瓶颈已定位为 StreamVGGT predicted depth/K。** GT camera geometry 和
   GT-depth+GT-K 能稳定工作；predicted depth 即使配 GT K 也失败，predicted K 也会独立伤害
   位姿。全局 scale、affine 和 Sim(3) 均不能稳定修复。
5. **正确的实例内二维 correspondence 可以显著修正相对位姿。** V9 A0-H 在 short、medium、
   long 三折全部通过，rotation 与 translation direction 同时改善，且没有恶化帧。
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

## 3. V7.4：动态实例与严格时间因果

### 3.1 目的

V7.3 已证明各种 transport branch 都能拟合长训练序列。V7.4 不再问容量，而是问：在动态实例、
严格未来帧和 parameter-matched control 下，SAM descriptor 是否提供超出 geometry 的可归因收益。

### 3.2 分支

- `uniform_transport`；
- `geometry_transport`；
- `sam_transport`；
- `sam_geometry_transport`；
- `sam_geometry_train_sam_off`：结构和参数量与主候选一致，但训练时禁用 SAM validity/logits。

主候选还运行 `sam_off`、`uniform_sam`、`wrong_sam_identity`、`shuffle_sam_time`。

### 3.3 结果

| fold | SAM+geometry vs L0 | vs geometry | vs trained-SAM-off | pass |
|---|---:|---:|---:|---:|
| short | +22.81% | -5.58% | -5.86% | 0 |
| medium | +10.26% | -2.52% | +8.29% | 0 |
| long | +22.01% | +9.21% | +3.29% | 0 |

long fold 虽然超过两个控制，但 `sam_off / wrong-ID / shuffle-time` 的 damage 约为
`-0.88% / -0.88% / -0.67%`，即破坏正确 SAM 输入反而略好。三个 fold 和 all-fold SAM causal
pass 均为 0。

### 3.4 本阶段证据

**证实：**

- 动态 birth、永久 slot、严格因果 memory 和无成熟实例 exact L0 fallback 已实现；
- SAM/geometry transport 具有训练容量；
- 同一序列训练帧 loss 可被压低。

**证伪或未通过：**

- 未证实 SAM descriptor 在未来帧提供超出 geometry 的稳定收益；
- 输入破坏没有形成稳定伤害，因此不能把 pose gain 归因于正确 SAM 内容；
- “训练到零”不能作为 SAM token 学到物理 correspondence 的证据。

## 4. V7.5：SAM region/identity 的显式约束

### 4.1 目的与方法

V7.5 完全不读取 SAM local descriptor，也不训练 pose head。它只测试 mask region 与 persistent ID：

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

## 5. V8 matcher-first：拆掉 direct pose shortcut

### 5.1 目的

V7 的问题是 matcher 和 pose head 被同一个末端 pose loss 驱动，head 可以绕过 correspondence。
V8 将其拆成：

1. Stage A：用 GT-world pseudo-match 单独监督 `p_ij`；
2. 冻结 matcher；
3. Stage B：evidence-only pose，不读取 camera hidden；
4. geometry、trained-SAM-off、no-match-supervision、dual Value 等控制共享相同 support。

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

## 6. V8 O1–O2.7：三维显式几何因子分解

这一组实验不训练网络，固定 GT-world pseudo-correspondence，逐层替换 geometry、depth、K 和
history pose，判断三维路线到底在哪里失效。

### 6.1 O1/O2.5：support 与 solver 上界

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

### 6.2 O2.6：depth、intrinsics、calibration、Sim(3)

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

### 6.3 O2.7：固定 K 与 SAM-instance depth affine

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

## 7. V9：二维 correspondence → 固定极几何

### 7.1 为什么转向二维路线

V8 已经证明 predicted depth/K 是三维显式求解的瓶颈。V9 不再使用 predicted depth、pointmap
Kabsch 或 learned SE(3) head，而是验证：

```text
instance-local 2D correspondence
  → calibrated essential/epipolar solver
  → relative rotation + translation direction
```

主实验使用 ScanNet++ calibrated K。它是传感器标定，在本场景中合理，但对于无标定互联网图片
仍属于 oracle deployment 条件。

### 7.2 Stage O 初版：dense visible-surface oracle

GT mesh/depth/pose 只用于构造同一可见 surface point 的 2D label。初版 solver 结果：

| fold | rotation gain | direction gain | worse frames | pass |
|---|---:|---:|---:|---:|
| short | +17.54% | +27.80% | 1 | 0 |
| medium | -338.38% | -623.51% | 3 | 0 |
| long | +18.82% | -230.40% | 2 | 0 |

所有 frame 均有大量 visible correspondence，因此失败不是 coverage，而是 eight-point
initialization 与无限 history-center fusion 的求解问题。

### 7.3 O-R1：固定 solver 修正

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

### 7.4 A0 all-edges：local32 current query oracle

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

### 7.5 A0-H：best-condition single history

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

### 7.6 A1：显式 SAM local matcher

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

## 8. 本周已经证实的内容

以下结论在当前 scene/protocol 下有直接实验支持：

1. **动态实例不需要固定物体参考帧。** 中途 birth、永久 slot、因果 memory 和第二次观测生效均已
   真实运行。
2. **learned pose head/transport 具有过拟合容量。** 这只能证明容量，不能证明 SAM 因果贡献。
3. **SAM mask/ID 是有信息的区域约束。** V7.5 局部 fold 和 O2.7 instance-affine 上界均支持这一
   点，但强度不足以形成稳定 all-fold pose 方法。
4. **GT 3D geometry + 正确坐标 + Kabsch 能工作。** 因此三维显式路线的失败不能再笼统归因于
   solver 实现错误。
5. **predicted depth 和 predicted K 都是独立瓶颈。** GT depth + GT K 通过；交叉替换任一预测量
   都会失败。
6. **predicted depth 误差不是单一 scale/offset。** scale、affine 与 Sim(3) 均没有全折通过。
7. **正确 instance-local 2D correspondence 能改善相对位姿。** O-R1 和 A0-H 均在三折形成明确
   上界。
8. **history edge 的可观测性很重要。** 在正确 correspondence 下，design condition 能避开
   450/495 的退化/错误 history edge。

## 9. 本周已经证伪或严格未通过的内容

“证伪”均限定于本周固定的数据、模型输入和实验协议，不代表对所有 SAM/StreamVGGT 方案的
普遍数学否定。

1. **训练 loss 接近零足以证明 SAM token 有用：证伪。** uniform、geometry、错误输入都能过拟合。
2. **V7.4 SAM+geometry 的未来帧收益来自正确 SAM descriptor：未通过。** controls 和扰动不支持
   因果归因。
3. **正确 SAM mask/ID history 能稳定改善所有未来 fold：未通过。** V7.5 all-fold region/history
   pass 全为 0。
4. **双 Value 或更丰富 fusion evidence 自动解决问题：未通过。** V8.1 dual Value 不稳定。
5. **只要显式监督 matcher 就能消除 pose shortcut：结构上成立，但性能命题未通过。** no-match
   control 可优于 supervised 分支。
6. **当前 predicted depth/K 可以直接支撑实例 Kabsch pose：证伪。** 即使对应为 GT oracle 也失败。
7. **固定 predicted K、全局 affine、实例 affine 或 Sim(3) 能稳定修复 predicted geometry：证伪。
   **所有对应分支均无 all-fold pass。
8. **所有 history edge 求均值会更稳定：证伪。** 450/495 的坏 edge 会压倒好 edge。
9. **当前 detector FPN local token 能通过 Linear Q/K 学出未来点对应：证伪。** held-out PCK
   0%–2.59%，所有 active pose frame 恶化。
10. **当前 A1 的 pose 变化对正确 SAM 内容存在稳定因果依赖：证伪。** channel permutation、
    wrong-ID、shuffle-time 没有形成预期的全折伤害。

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
