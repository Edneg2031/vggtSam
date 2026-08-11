# SAM3.1 × StreamVGGT 实验结论

> 时间：2026-08-03 至 2026-08-11
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

## 8. V9.1–V9.4 新证据与终止结论

V9.1 已完成，不再重复：

| fold | actual local32 key coverage | 离散 GT PCK | trained SAM Q/K future PCK | pose 结果 |
|---|---:|---:|---:|---|
| short | 80.31% | 54.92% | 6.74% | 恶化 |
| medium | 44.44% | 25.93% | 0% | 恶化 |
| long | 47.79% | 27.71% | 1.20% | 恶化 |

这证伪了“连续 GT local32 上界通过，所以实际 local32 history key 也足够”。实际离散 key 的覆盖、
PCK 和 pose 都未通过；训练 Q/K 的未来表现又明显低于离散上界，说明 **support 稀疏**和
**descriptor/QK 时间外推失败**是两个不同问题。

V9.2 进一步固定 current query32，使用同一个 farthest-UV cache 的 32/64/128/256 history key
嵌套前缀，并比较 nearest、mutual、greedy-unique：

| history keys | mean coverage@12 | mean nearest PCK@8 | all-fold pose pass |
|---:|---:|---:|---:|
| 32 | 57.52% | 36.19% | 0 |
| 64 | 77.78% | 55.90% | 0 |
| 128 | 95.13% | 74.98% | 0 |
| 256 | 99.45% | 91.68% | 0 |

mutual/greedy-unique 消除输出重复 key 后仍全部失败。这证伪了“history token 太少”以及“重复
key 碰撞”是最终位姿瓶颈。PCK 提高而 pose 继续恶化，说明 PCK@8 对 localized-instance
essential solver 过宽；当前 3–5px 离散量化误差或空间条件仍可能破坏 pose。

V9.3 在每个测试帧冻结相同 history edge 后得到：

| correspondence | selected EPE | mean R gain | mean t-dir gain | 恶化行 | all-fold pass |
|---|---:|---:|---:|---:|---:|
| continuous GT | 0px | 77.20% | 85.52% | 0 | 1 |
| hard nearest-256 | 4.07px | -222.92% | -35.36% | 7 | 0 |
| GT soft-convex K8 | 0.65px | 23.13% | 58.81% | 3 | 0 |
| continuous + 0.5px noise | 0.63px | -82.32% | 10.18% | 7 | 0 |

continuous 正对照三折通过，排除了此前收益只是分支选择不同导致的假象。soft-K8 虽然平均改善，
仍有逐帧恶化；0.5px 噪声已经无法三折通过。EPE≤1/2px 的 hard-match 子集只保留 6%/17%，
不足以稳定求解。这说明 localized-instance O-R1 对亚像素误差非常敏感，继续训练 matcher 前必须
先证明存在能承受该误差的鲁棒 solver。

V9.4 固定 V9.3 的 evidence/history edge，比较 O-R1、确定性 RANSAC minimal、RANSAC
inlier-refine 和空间均衡 RANSAC：

| evidence | 最好现象 | all-fold pass |
|---|---|---:|
| continuous GT | O-R1 与 deterministic RANSAC minimal 三折通过、零恶化 | 1 |
| GT soft-convex K8 | spatial RANSAC 只剩 1 个恶化行，但存在 inactive 且 medium 不稳 | 0 |
| continuous + 0.5px noise | spatial RANSAC 平均 R/t-dir 改善，但有 6 个恶化行 | 0 |
| continuous + 1.0px noise | 所有 solver 均不稳定 | 0 |

continuous 正对照通过说明协议、固定边和 solver 实现有效；soft-K8 与 0.5px noise 在所有 robust
solver 下失败，满足预设终止条件。因此正式停止
`localized SAM-instance correspondence → essential matrix → pose` 路线，不再调整该路线的
detector-FPN descriptor、token 数、Q/K、soft decoder 或 RANSAC 参数。

这不代表所有 SAM3.1 feature 无效；它只证伪了当前局部实例 correspondence 可以通过固定
essential solver 稳定修正 pose。

## 9. V9.5：空间支撑上界

V9.5 不再优化 localized-instance solver，而是直接检验 V9.4 失败是否由实例区域过于局部造成：

- 固定 V9.3/V9.4 的同一组 12 条 history edge；
- 固定 O-R1 solver，不训练 matcher 或 pose model；
- 比较 `instance-local32`、`full-image`、`instance + background 50/50`；
- 每条边严格配平到相同 correspondence 数，排除“点更多”的混杂因素；
- 在 support 选定后分别加入 0、0.5、1.0px 噪声，噪声分支运行 5 个固定 seed replicate；
- GT joint-view 选点是诊断上界，任何通过都不能直接归因给 SAM descriptor。

V9.5 结果三折全部配平到平均 26.25 条 correspondence：

| support | noise | current hull | mean R gain | mean t-dir gain | 恶化 | pass |
|---|---:|---:|---:|---:|---:|---:|
| instance-local32 | 0px | 0.130 | 77.20% | 85.52% | 0 | 1 |
| instance-local32 | 0.5px | 0.130 | -52.61% | -6.59% | 10 | 0 |
| full-image | 0.5px | 0.576 | 83.76% | 89.68% | 0 | 1 |
| full-image | 1.0px | 0.576 | 74.63% | 80.89% | 0 | 1 |
| instance/background balanced | 0.5px | 0.566 | 81.93% | 87.96% | 0 | 1 |
| instance/background balanced | 1.0px | 0.566 | 73.26% | 81.54% | 0 | 1 |

这把 V9.4 的失败定位为**局部实例区域空间条件不足**，而不是 O-R1 本身无法容忍 0.5–1px
误差。等量全图或实例/背景混合支撑都能稳定承受 1px，说明 fixed-essential 路线只能转为
“全局匹配 + SAM 分层”，不能继续使用 localized-instance-only correspondence。

它仍未证明 SAM 有用：full-image 略好于 balanced，且所有 correspondence 都来自 GT 上界。

## 10. V9.6：实际 72×72 网格坐标 gate

V9.6 在训练 dense matcher 之前验证真实离散坐标是否具有可行上界：

- current query 必须来自实际 SAM `detector_fpn2` 的 72×72 align-corners 网格；
- support 选择只读取 current UV 和 current mask，不读取 history GT UV 或 GT pose error；
- history 比较 continuous GT、单个实际 grid key hard-nearest、四个相邻实际 key 的
  soft-bilinear-K4；
- 空间策略比较 full-grid、SAM-mask balanced、bbox balanced、等面积随机移位 mask balanced；
- 每条边仍配平到 V9.5 instance-local32 的相同数量，固定同一组 12 条 history edge 和 O-R1；
- PCK 收紧到 1px；不缓存或评价 dense descriptor，也不训练任何模型。

V9.6 的主 gate 只回答真实 72×72 网格能否表达可求解的连续坐标，因此由 full-grid continuous
和 full-grid soft-K4 决定。SAM/bbox/random balanced 只诊断区域选择；某条边无法严格凑齐 50/50
时不得反向否决 full-grid 坐标 gate，也不得被算成 SAM 胜出。

若 bbox 或随机 mask 在某条边无法提供足够的 region/complement 点，runner 会保留可用点、标记
`equal-count feasible=0` 和 `pass=0`，但继续执行主 gate；不可构造的负控制不会被算成 SAM 胜出。

一键命令：`zsh streaming_couping/commands_v96_dense_grid_upper_bound.txt`。

实际结果：full-grid continuous 与 soft-K4 均三折通过、零恶化；hard-nearest 失败（平均
EPE 2.34px）。因此已证实真实网格需要子网格解码，并已允许进入 dense descriptor 实验。
SAM-balanced soft-K4 本身零恶化，但 long fold 无法严格配平，不能据此声明 SAM mask 独有贡献。

## 11. V9.7：dense detector-FPN 已证伪

V9.7 不再重复 local32、token 数、RANSAC、depth/K 或 mask-support 上界：

- 每帧一次性缓存完整 SAM3.1 `detector_fpn2` 72×72 descriptor；
- current query 使用 current-only FPS，history 使用全部 5184 个真实 grid key；
- matcher 只预测 coarse history cell 和 bounded 2D sub-cell offset；
- 训练只使用 GT 2D correspondence label，不使用 pose loss；
- matcher 冻结后，用固定 V9.3 history edge 和 O-R1 评价未来 short/medium/long；
- 参数与 support 完全匹配的控制为 StreamVGGT patch、coordinate-only、SAM train-off；
- 正常 SAM 另做 SAM-off、channel permutation、shuffle-time 扰乱。

这里 SAM 的作用路径是：**SAM dense descriptor 预测当前点在历史帧的连续二维位置，固定极几何
solver 再用这些对应修正 StreamVGGT L0 的相对旋转和位移方向。** 它不把 SAM token 注入
StreamVGGT backbone，也没有可直接背 pose 的 adapter。

实际结果：

| fold | train loss | future PCK@1 | future EPE | pose 结果 |
|---|---:|---:|---:|---|
| short | 10.29 → 0.38 | 0 | 110.69px | 4/4 恶化 |
| medium | 9.97 → 1.15 | 0 | 98.00px | 3/4 恶化，1 帧 inactive 回退 |
| long | 10.03 → 2.00 | 1.65% | 105.22px | 4/4 恶化 |

train-off loss 不下降，说明正常 matcher 的梯度和 descriptor 输入确实生效；但未来 correspondence
几乎为零，StreamVGGT dense control 的 EPE 也始终低于 SAM。因此证伪的是：

> 逐帧 `detector_fpn2` dense descriptor 虽可拟合训练前缀，但不能在当前同场景时间外推协议中
> 预测未来 correspondence，也不能帮助 StreamVGGT relative pose。

V9.6 的 full-grid soft-K4 上界已经通过，所以该失败不能再归因于 72×72 坐标量化、局部实例
support 或 O-R1 的 1px 容差。它不代表 SAM3.1 video memory feature 无效。

一键命令：`zsh streaming_couping/commands_v97_dense_descriptor_causality.txt`。

## 12. V9.8：只验证真正的 SAM3.1 temporal memory

V9.8 是 V9.7 后唯一新增假设，不再调整 detector-FPN、depth/K、token 数或 solver：

- 在 SAM3.1 multiplex tracker 的 `_prepare_memory_conditioned_features` 返回处捕获特征；这是当前帧
  propagation feature 已读取过去 `maskmem_features` 和 `obj_ptr` 后的真实 history-read map；
- memory-off 是**同一次调用**进入 memory encoder 前的 raw propagation map，不用另一模型近似；
- `bed`、`wardrobe` 各自运行 forward-only session，再对可用 prompt map 等权合并；中途新实例可由
  tracker 动态 birth，并从后续帧的 memory read 开始参与；
- multiplex 输出按 bucket joint feature 处理，不错误地当成 per-object tensor 做 `demux`；
- query/key、5184 history keys、V9.3 fixed edges、O-R1 和训练 label 与 V9.7 完全相同；
- matcher seed、初始化、batch 顺序和优化器也锁定为 V9.7 的设置，不重训已经完成的 detector-FPN；
  完全相同的 V9.7 zero-input checkpoint 也直接复用，不重复训练；
- 正式控制：same-call memory-off、V9.7 detector-FPN、StreamVGGT、train-off；扰乱为 memory-off-at-eval、
  channel permutation 和 causal time shuffle；
- 先从 V9.7 checkpoint 输出 train/future PCK/EPE 审计，再训练 V9.8 correspondence matcher；不训练
  pose model，也不使用 pose loss。

只有 temporal feature 在三个未来 fold 都同时满足以下条件，才能写“SAM3.1 memory 帮助
StreamVGGT relative pose”：

1. PCK@1 更高、EPE 更低，超过全部正式控制；
2. memory-off/time-shuffle/channel-permute 都破坏 correspondence；
3. 固定 O-R1 的 rotation 与 translation-direction 都优于 L0，且没有恶化帧；
4. pose 也优于全部控制，三折同时通过。

训练 loss 下降、训练 PCK 高或单折 pose 变好都不算成功。该实验仍只允许同场景时间外推的
relative pose 结论。

一键命令：`zsh streaming_couping/commands_v98_temporal_memory_causality.txt`。
