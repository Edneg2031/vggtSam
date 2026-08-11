# SAM3.1 × StreamVGGT 实验结论

> 时间：2026-08-03 至 2026-08-06
>
> 目标：验证 SAM3.1 的 mask、persistent identity 和 local token 是否能帮助 StreamVGGT 修正相机位姿。

## 1. 一句话结论

**已经证实：正确的实例内二维跨帧对应可以通过固定极几何求解器明显改善 StreamVGGT 的相对
旋转和位移方向。**

**尚未证实：这些正确对应可以由当前 SAM3.1 `detector_fpn2` local token 产生。当前测试的
SAM-token adapter、feature transport 和 Linear Q/K matcher 都没有形成稳定的未来帧因果收益。**

## 2. 实验设置

### 2.1 数据

- 数据集：ScanNet++；
- scene：`00a231a370`；
- 序列：frame 90 至 525，每隔 15 帧取一帧，共 30 帧；
- frame 90 只定义 camera gauge，不要求物体必须在该帧出现；
- 所有正式结果都是**同一场景内时间外推**，不是跨场景泛化。

主要严格时间划分：

| fold | 训练帧 | 未来测试帧 |
|---|---|---|
| short | 270–345，step 15 | 360、375、390、405 |
| medium | 270–405，step 15 | 420、435、450、465 |
| long | 270–465，step 15 | 480、495、510、525 |

### 2.2 SAM3.1

- discovery prompts：`bed`、`wardrobe`；
- forward-only 检测和跟踪；
- 最多 8 个永久 slot；
- 实际发现 3 条主要 track：两个在 frame 90 出生，一个 bed 在 frame 150 出生；
- 新实例出生帧只写入 memory，从下一次可靠观测开始才能参与位姿；
- mask 用于限定实例区域，persistent ID 用于关联当前与历史实例；
- local feature：`detector_fpn2`，72×72 grid，每实例最多 32 个 token，farthest-UV sampling；
- 当前 local descriptor 是逐图像 detector feature，不是 SAM3.1 video-memory temporal feature。

### 2.3 StreamVGGT 与评价边界

- StreamVGGT backbone 冻结；
- learned 实验从 frozen camera L0 开始预测 pose residual；
- 显式 solver 实验不训练 pose model；
- matching 使用 PCK@8px、EPE 和 Sampson RMSE；
- 最终二维方法只评价 relative rotation 和 translation direction，不支持 metric translation scale、
  absolute center 或 absolute trajectory 结论。

## 3. 已测试的 SAM → StreamVGGT 信息路径

| SAM 信息 | 如何进入 StreamVGGT | 如何修改 pose |
|---|---|---|
| pooled/global token | 与 camera hidden、geometry 拼接 | learned adapter 直接输出 SE(3) residual |
| local descriptor 作为 affinity | SAM current/history Q/K 产生权重，搬运历史 StreamVGGT geometry | pooled evidence 进入 SE(3) adapter |
| local descriptor 作为 affinity + Value | 同时搬运历史 geometry 和 SAM descriptor | evidence-only adapter 输出 SE(3) residual |
| mask + persistent ID | mask 选择 StreamVGGT points，ID 建立历史 object map | ICP、ray-centre 或 Kabsch 显式求解 |
| local descriptor 作为二维 matcher | SAM current/history local32 预测 2D correspondence | fixed essential/epipolar solver 修正相对 pose |

## 4. 什么信息被证实有用

### 4.1 GT 三维信息的有效上界

| 使用的信息 | 结果 | 说明 |
|---|---:|---|
| GT camera geometry + GT history + Kabsch | 三折通过 | 坐标转换、support 和显式三维 solver 可以工作 |
| GT depth + GT intrinsics K | 三折通过 | depth backprojection 和 solver 实现正确 |
| GT depth + predicted K | 失败 | predicted intrinsics 会独立破坏位姿 |
| predicted depth + GT K | 失败 | predicted depth 会独立破坏位姿 |
| predicted depth + predicted K | 失败 | 当前 StreamVGGT 预测几何不足以支撑实例 Kabsch |

这说明三维路线失败的主要原因不是“对应点数量不够”或“Kabsch 写错”，而是 predicted depth 的
局部 shape/跨帧一致性和 predicted K。

### 4.2 GT 二维对应的有效上界

使用 GT mesh/depth/pose 只生成正确的可见表面二维 correspondence，再交给固定求解器：

| fold | rotation gain | translation-direction gain | 恶化帧 |
|---|---:|---:|---:|
| short | 75.65% | 92.00% | 0 |
| medium | 32.90% | 86.99% | 0 |
| long | 78.37% | 58.33% | 0 |

结论：**正确二维 correspondence 确实能修正 StreamVGGT relative pose。**

### 4.3 local32 query 的连续 GT 对应上界

固定实际 current local32 query，但用 GT 将其连续投影到 history image，并选择 design condition
最好的因果 history edge：

| fold | raw aggregate | refined aggregate | gain | 恶化帧 |
|---|---:|---:|---:|---:|
| short | 13.737° | 1.279° | 90.69% | 0 |
| medium | 13.801° | 1.707° | 87.63% | 0 |
| long | 25.748° | 6.877° | 73.29% | 0 |

这证明 current local32 的 query 位置和固定 solver 存在可行上界。但 history 对应是连续 GT UV，
不是从实际 32 个 history token 中匹配出来的，因此这仍然不是 SAM-token 成功证据。

### 4.4 SAM mask/ID 的有限作用

- 动态 birth、永久 slot、因果 memory 和多实例参与流程已经正常工作；
- mask 定义的实例区域在局部 fold 中优于 full image、bbox、random 和 stale mask；
- mask 内的 instance-wise depth affine 上界优于全图 affine。

但这些收益没有稳定通过所有时间 fold，因此只能说明 mask/ID 包含有意义的区域和身份信息，
不能说明已经形成稳定 pose 方法，更不能说明 SAM appearance token 有效。

## 5. 什么方法已经测试但无效

| 已测试方法 | 实际结果 | 基于结果的结论 |
|---|---|---|
| pooled/global SAM token + camera/geometry → direct pose adapter | 训练或部分 held-out 可以下降，但输入贡献无法隔离 | pipeline 有拟合能力，不能归因给 SAM token |
| SAM local Q/K → 搬运 StreamVGGT geometry → SE(3) adapter | 三个未来 fold causal pass 全为 0；wrong-ID/shuffle 后没有稳定变差 | SAM affinity 没有建立稳定因果收益 |
| SAM + geometry dual Value → evidence-only adapter | short/medium 下降，long 仍差于 L0；无 all-fold pass | 增加 SAM Value 或更丰富 evidence 不能解决问题 |
| GT matching supervision → 冻结 matcher → evidence-only pose | no-match-supervision control 可以更好；SAM perturbation 不稳定 | 显式监督结构排除了部分捷径，但没有证明 SAM token 有用 |
| SAM mask/ID points → ICP/history map/ray solver | 只在 short/frozen 局部通过，medium/long 不稳定 | mask/ID object map 不是稳定 pose 修正器 |
| predicted depth/K + GT correspondence → Kabsch | 三折失败 | 即使对应正确，当前 predicted 3D geometry 仍是瓶颈 |
| fixed/median K、scale、global/instance affine depth、Sim(3) | 全部没有 all-fold pass | predicted geometry 不是简单 scale/offset 问题 |
| 所有 history edge 求均值 | frame 450/495 被坏 edge 拉差 | history 越多不一定越好，必须考虑可观测性 |
| `detector_fpn2 local32 → Linear Q/K → 2D matcher` | train loss 下降 85%–92%，但未来 PCK 仅 0%–2.59% | loss 下降不等于学到 correspondence |

当前 SAM matcher 的具体结果：

| fold | train loss | future PCK | EPE | raw pose aggregate | refined pose aggregate | 恶化帧 |
|---|---:|---:|---:|---:|---:|---:|
| short | 5.153 → 0.429 | 2.59% | 142.37px | 17.886° | 128.781° | 4/4 |
| medium | 4.452 → 0.548 | 0% | 284.45px | 19.761° | 111.955° | 4/4 |
| long | 4.098 → 0.595 | 0.80% | 321.93px | 25.988° | 154.722° | 4/4 |

正常 SAM 没有稳定超过 StreamVGGT patch、trained-off、wrong-ID、shuffle-time 或 channel
permutation；所有 12 个未来测试帧的 pose 都恶化。因此，当前
`detector_fpn2 + local32 + Linear Q/K` 路线在本协议下无效。

## 6. adapter 过拟合应如何解释

正确结论：

> learned adapter 有足够容量拟合当前序列的 pose residual。

不能得出的结论：

> 带 SAM 输入的 adapter 把训练 loss 降到零，所以 SAM token 能修正位姿。

uniform transport、geometry-only、SAM-off 和错误 SAM 输入同样可以过拟合。有效的是 adapter 的
参数容量，而不是 SAM token。只有正常 SAM 在严格未来帧超过参数匹配控制，并且 wrong-ID、
shuffle-time、channel permutation 稳定破坏 matching 和 pose，才能形成 SAM-token 因果证据。

## 7. 当前可以和不可以对外表述的结论

可以表述：

1. SAM3.1 动态 mask/ID 可以支持中途实例出生和多实例因果 memory；
2. 正确的实例内 2D correspondence 可以显著改善 StreamVGGT relative pose；
3. 当前显式三维路线的主要瓶颈是 StreamVGGT predicted depth/K；
4. 当前 `detector_fpn2` local token matcher 没有学出未来 correspondence。

不可以表述：

1. SAM token 已经提升了 StreamVGGT pose；
2. adapter 训练到零证明了 SAM token 有效；
3. 同场景时间外推等于跨场景泛化；
4. 二维 essential solver 改善了 metric center 或 absolute trajectory；
5. detector-FPN 失败代表所有 SAM3.1 temporal feature 都无效。

## 8. V9.1 新证据与下一步

V9.1 已完成，不再重复：

| fold | actual local32 key coverage | 离散 GT PCK | trained SAM Q/K future PCK | pose 结果 |
|---|---:|---:|---:|---|
| short | 80.31% | 54.92% | 6.74% | 恶化 |
| medium | 44.44% | 25.93% | 0% | 恶化 |
| long | 47.79% | 27.71% | 1.20% | 恶化 |

这证伪了“连续 GT local32 上界通过，所以实际 local32 history key 也足够”。实际离散 key 的覆盖、
PCK 和 pose 都未通过；训练 Q/K 的未来表现又明显低于离散上界，说明 **support 稀疏**和
**descriptor/QK 时间外推失败**是两个不同问题。

下一步固定 current query32、动态多实例和 O-R1 solver，只做 V9.2 support 因子分解：

- history key 使用同一个 farthest-UV 256-token cache 的 32/64/128/256 嵌套前缀；
- 比较 nearest、mutual、greedy-unique，区分稀疏覆盖与重复 key 碰撞；
- 不训练 matcher，不增加 prompt，不增加 pose adapter；
- 只有离散 GT 上界三折通过后，才测试新的 SAM3.1 temporal memory feature。

一键命令：`zsh streaming_couping/commands_v92_support_factorization.txt`。
