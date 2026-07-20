# 流式几何重建 × 开放词表实例跟踪 耦合系统技术方案细化

> **研究记录。** 本文后半部分保留已经尝试过的设计、负结果和后续设想，其中
> 多项实验入口已从代码中删除。当前已验证主线和命令以
> [`../readme.md`](../readme.md) 与 [`../commands.txt`](../commands.txt) 为准。

## 当前已验证方法（2026-07-20）

当前方法不是 token fusion，也不修改 SAM3 decoder。它使用显式 2D–3D 几何作为
冻结 SAM3 与冻结 StreamVGGT 之间的桥：

```text
SAM3 mask + persistent obj_id
  -> 与历史实例 3D 支持计算 tracker geometry coverage
  -> coverage < 0.50 时判断可能发生高置信跟错
  -> SAM3 global-text 生成当前帧完整同类候选
  -> 时序对齐几何按 support coverage 选择历史实例
  -> support coverage >= 0.25 时写回原 obj_id memory
  -> 恢复未来 mask，并用可靠 mask 生成实例点云
```

对象地图只接收 score 与 geometry coverage 均可靠的 tracking observation；
`map_update_min_geometry_coverage=0.50`。reference 后的 GT mask/pointmap 只进入
指标和显式 control，不进入 deployable 方法。

在场景 `00a231a370`、帧 `90 105 119 130 140 210 240` 的 bed(54) 压力测试中：

- 原始帧 119：SAM3 score `0.9844`、IoU `0.0207`；
- geometry-selected candidate：IoU `0.9323`，等于 candidate oracle；
- no-memory 后续 4 帧平均 IoU：`0.0002`；
- same-ID memory 后续 4 帧平均 IoU：`0.7763`；
- object-map F-score@10cm：`0.0819 -> 0.9443`；
- GT-mask map oracle F-score@10cm：`0.9481`；
- shuffled geometry 无恢复；37/68 natural gate 无误触发。

由此已经验证“geometry → tracking recovery → better instance point cloud”。历史
tracking 扩充的 object map 与 reference-only map 在本例中选中同一候选，因而
尚不能声称历史扩图进一步改善了恢复选择。

`oracle_mask_same_id_memory` 是 GT 当前可见 mask 写回控制，不是未来传播上限；
完整 text candidate 的未来追踪和地图结果更好。

本阶段现已暂停，不继续 held-out 或阈值调优。下一阶段研究实例点云与相机位姿：

1. 以 `instance_point_cloud.py` 和 `instance_map_evaluation.py` 为点云基线；
2. 核对 StreamVGGT `world_to_camera`、pointmap 与 GT pose 的坐标约定；
3. 建立 raw StreamVGGT pose 的 ATE/RPE 与逐帧点云一致性诊断；
4. 再判断可靠静态实例是否能形成相机位姿约束；
5. 旧 translation-only ICP 仅作为历史诊断，不直接恢复为默认方案。

其中第 2、3 项已实现于 `pose_pointmap_diagnostics.py`。raw baseline 表明：

- reference-pose 对齐后的 ATE RMSE 为 `0.2280 m`；
- all-pairs rotation mean 为 `2.39°`，rotation@5° 为 `100%`；
- translation-direction mean 为 `14.56°`，@5° 仅 `19.0%`；
- `105->119` 方向误差 `34.33°`，GT/预测位移为 `0.260/0.300 m`；
- `210->240` 方向误差 `44.22°`，GT/预测位移为 `0.123/0.312 m`；
- non-reference pointmap 平均 frame RMSE 为 `0.1329 m`，帧 240 为
  `0.1866 m`，低于同帧 camera-center 误差 `0.460 m`；
- predicted focal 全部偏大，但 focal 误差与逐帧 pointmap RMSE 相关性很低，
  因而不是 pointmap 漂移的主要解释。

这排除了单一全局 scale 和纯累积漂移解释。当前第一项位姿修复不是旧实例 ICP，
而是 pointmap-consistent ray-center：固定 `R/K`，由每个世界点和对应像素射线
线性重估 camera center，再以 `t=-RC` 更新 StreamVGGT world-to-camera。
同一次服务器运行包含 predicted-K 可部署分支、GT K/R oracle、trimming 消融和
spatially-shuffled pointmap 负对照；设计说明见
[`pose_pointmap_diagnostics.md`](pose_pointmap_diagnostics.md)。

## 0. 总体框架

```
输入: 视频/图像序列 I_1..I_t (流式到达) + 文本 prompt (概念/目标描述)
        │                                   │
        ▼                                   ▼
┌───────────────────┐              ┌────────────────────┐
│  几何分支 (StreamVGGT)│              │  跟踪分支 (SAM3)      │
│  causal KV cache     │◄──桥接模块──►│  dense memory bank   │
│  → 相机位姿 P_t        │  Bridge      │  → 实例mask M_t, ID    │
│  → 深度/点云 D_t        │              │  → 置信度 c_t^k        │
└───────────────────┘              └────────────────────┘
        │                                   │
        └───────────► 融合模块 ◄─────────────┘
                       │
                       ▼
        实例一致mask + 物体级语义点云地图 {O_k: pts, label, traj}
```

核心设计原则：**两个主干不做权重共享**。主要交互仍通过 (a) 相机约束、
(b) 跟踪增强、(c) 置信度门控三个显式接口；§3.1 额外允许一个可单独消融的
低分辨率 adapter，但不能替代显式几何对齐或绕过 SAM3 原始 gate。

---

## 1. 桥接模块 (Cross-Modal Bridge)

两个分支的 token 空间不兼容（几何 token 编码深度/相机隐状态，SAM3 memory
token 编码外观/语义）。因此默认使用**显式几何量作为中介**；§3.1 的特征
融合只发生在独立轻量 adapter 中，并必须通过 aligned/shuffled 对照证明它确实
消费了时序正确的几何：

- 几何分支只对外暴露：相机位姿 `P_t (R_t, T_t)`、逐像素深度 `D_t`、可选的逐点置信度 `σ_t`（StreamVGGT/VGGT 本身会输出置信度图，直接复用）。
- 跟踪分支只对外暴露：实例 mask `M_t^k`、实例 ID、跟踪置信度 `c_t^k`（可以用 SAM3 的 objectness/matching score）。
- 桥接模块是一个轻量模块，职责是把这两组显式量互相转换（2D↔3D 投影、加权残差构造），不承担表征学习任务，因此**可以不联合训练几何/跟踪两个主干**，只训练桥接模块本身（残差网络或简单的可微分几何变换），大幅降低训练成本和过拟合风险。

最关键的设计选择仍是让显式几何变换承担空间对齐，学习模块只负责把可信的
几何证据转成 SAM3 可消费的 residual，而不让两个主干黑箱式联合训练。

---

## 2. 相机约束的具体公式化

### 2.1 实例级回环残差（动态场景也适用）

当 SAM3 判定实例 k 在帧 i 和帧 j（j >> i，大视角跨度）为同一实例，且两帧的跟踪置信度都高于阈值时，取该实例在两帧的 mask 内的几何分支深度点：

```
X_i^k = π^-1(D_i, M_i^k, P_i)   # 反投影到世界系
X_j^k = π^-1(D_j, M_j^k, P_j)
```

如果该实例被判定为**刚体/静止**（用几何分支自身的场景流残差或速度估计判断），构造重投影残差：

```
L_loop = Σ_k Σ_{p ∈ correspond(X_i^k, X_j^k)} ρ( π(X_i^k[p], P_j) - x_j^k[p] )
```

其中 `π` 是相机投影函数，`ρ` 是 robust loss（Huber），`correspond(·)` 用最近邻或者光流做点对应。这个残差以**软约束**的形式加入相机位姿更新（如果几何分支支持在线优化/滑窗BA，直接加入BA目标；如果是纯前馈causal模型不支持显式优化，则退化为对 KV cache 中相机相关 token 的一个辅助 loss，在训练时加入，推理时靠模型隐式利用）。

**关键判断点**：StreamVGGT 是纯前馈因果模型，本身没有显式的位姿优化环节，所以这个"约束"在推理时**不能**指望像传统SLAM那样做显式BA修正——除非你额外接一个轻量的滑窗位姿精修层（例如对最近 N 帧做一个可微分的几何一致性精修，只优化位姿不优化网络权重）。这一点建议明确写进方案：**约束在训练阶段体现为辅助loss，在推理阶段体现为一个轻量后处理精修模块**，两者分开设计。

### 2.2 静态物体先验（更稳，作为baseline约束）

对判定为静止的实例，其点云重心在世界系应近似不变：

```
L_static = Σ_k || centroid(X_i^k) - centroid(X_j^k) ||^2 ,  k ∈ StaticSet
```

这个约束更弱但更鲁棒，建议作为第一阶段实验的主约束，回环残差作为第二阶段加强项。

当前受控实现比较 `GT oracle / SAM3 original / hard-recovery memory` 三种实例
mask，并使用 StreamVGGT `point_head` 世界点云做 translation-only ICP。接受后的
`Delta T` 作用于整帧 pointmap；同步相机轨迹仅作为 proxy，不解释为严格相机
优化。`reference_only` 固定使用首帧实例点，`causal` 则在可靠 ICP 后把新表面
体素合入持久 object map。两者共享 mask、pointmap、Sim(3) 和 ICP 参数，用于
验证历史实例地图能否缓解首帧局部可见造成的配准偏差。
若 causal map 的实例指标改善但整场景指标下降，则增加无 GT 的 scene guard：
在实例 ICP 平移方向上离散测试多个阻尼系数，以历史高置信场景点的 trimmed
NN RMSE/fitness 为约束，选取不破坏全局重合度的最大增量。
当前单场景实验中该自一致性 guard 未产生阻尼，说明 point-head 场景可能整体
自洽但仍偏离真实坐标。该路线降级为诊断，先通过固定 alpha 消融量化单实例
约束的 Pareto 边界；若无法兼顾实例与全局误差，再进入多实例共享位姿优化。
该消融中 object map 始终用完整实例 ICP 更新，alpha 只阻尼整帧 pointmap 增量，
避免把“历史地图质量”和“全局修正强度”混成同一个变量。
当前联合使用 `cabinet(37) / wardrobe(68) / bed(54)`：三个实例分别维护 SAM3
memory 与 causal object map，但在每帧将通过门控的实例对应合并，实例等权求解
一个共享平移 `Delta T_t`，并只对整帧 pointmap 应用一次。至少两个实例通过门控，
且各实例建议平移的一致性满足阈值时才更新；不再叠加三个独立 ICP 增量。

---

## 3. 几何增强跟踪的具体机制

### 3.1 3D-aware Memory Warping

当前把 §3.1 实现为“显式位置对齐 + 轻量特征融合”，不降低 SAM3 的
object-presence 阈值：

```text
StreamVGGT aggregator layers 4/11/17: [T,2048,24,24]
        -> shallow self-attention
        -> sequential cross-attention with deeper layers
        -> merged geometry: [T,256,24,24]

SAM3 tracker FPN2: [T,256,72,72]
        + resize(merged geometry) + geometry confidence
        -> Conv fusion + learned confidence gate
        -> FPN2 residual: [T,256,72,72]
        -> original SAM3 memory attention -> mask decoder
        -> original object_score_logits > 0 gate -> memory encoder

历史 maskmem_pos_enc
        -> history pixel -> 3D -> current view reprojection
        -> geometry-warped position encoding
        -> original SAM3 memory attention
```

SAM3 与 StreamVGGT 均冻结，只训练多层几何 merger、卷积融合和 FPN2 residual
adapter。训练使用实例 mask 的 focal/Dice、可见性的 presence BCE，以及
`aligned > shuffled` 的几何排序损失。训练时使用 GT visibility teacher forcing
避免正样本在 gate 前被截断；最终推理关闭 teacher forcing，恢复 SAM3 原始 gate。

消融固定为 `original / warp_only / merger_only / complete / shuffled`。只有
`complete` 优于 `original`、且 aligned 优于 shuffled，才能说明时序正确的 3D
信息被 SAM3 使用，而不是 adapter 单纯记住该序列。

### 3.2 3D近邻 Fallback 检索

当外观特征检索置信度低于阈值（tracker 认为可能丢失目标）时，触发 fallback：用当前帧的深度点云在**历史3D点云地图**里做最近邻检索，找回候选区域，再用 SAM3 的 mask decoder 在候选区域做精细分割确认。这一步只在**低置信度时触发**，不影响正常情况下的效率。

---

## 4. 置信度门控（贯穿全系统的关键机制）

因为两个分支互相依赖对方的输出质量，必须有门控，否则误差会互相放大。建议统一的门控公式：

当前保留可工作的 hard-memory 路线，并改为逐实例循环更新，而不是只使用
reference frame 做一次回退：

```text
reference GT prompt
  -> 初始化该实例的 StreamVGGT 3D object map
  -> SAM3 原生逐帧跟踪
  -> 可靠 mask + 几何一致：把当前新视角点加入 object map
  -> mask 缺失/像素过少/面积塌缩：
       投影累计 object map 得到几何候选
       -> 全图文本候选由几何支持点选择
       -> 完整恢复 mask 写回同一 SAM3 obj_id 的 memory
  -> 后续继续跟踪、更新地图，并允许再次恢复
```

低 SAM3 score 不会单独否决 mask：若 mask 与时序对齐的几何支持区域一致，仍可
更新地图；只有低分且缺少几何一致性时才视为弱跟踪。恢复 mask 必须覆盖足够比例
的几何支持点。每个实例维护独立 `obj_id`、SAM3 memory 和 3D object map；多实例
联合 ICP 只共享最终帧平移，不会混合实例状态。除 reference prompt 外，后续 GT
mask 只用于指标，不参与门控、地图更新或恢复。

对于首帧仅局部可见的大物体，默认不用局部几何框定义分割范围：SAM3 先根据
全图文本生成完整实例候选，再以投影支持区域选择与历史实例对应的候选，最后把
完整候选 mask 写回同一 `obj_id`。旧的 `text_box_points` 保留为消融；正点只证明
“这些像素属于该实例”，本身不能表达床等物体的完整空间范围。

这里两种 memory 不应混淆：StreamVGGT 点只更新外部 3D object map；通过几何
找回的完整 2D mask 才编码进 SAM3 原生 memory。这样第二次可靠观察到的新表面会
扩充 3D 地图，后续大视角恢复不再永远受首帧局部可见范围限制。

下一步先单独消融“全图文本候选中选哪一个实例”，不同时修改 SAM3 decoder、
memory gate 或相机优化。历史中通过门控的实例 mask 在冻结的 StreamVGGT
aggregator layer 17 原生 patch 网格（当前序列为 `[T,2048,22,37]`）上做置信度
加权池化，形成累计实例描述子；
恢复帧的每个 SAM3 文本候选同样池化，再比较：

1. `geometry_only`：只按候选与投影支持区域的 IoU 选择，作为当前 hard baseline。
2. `descriptor_only`：只按候选与历史实例描述子的余弦相似度选择。
3. `geometry_descriptor`：几何 IoU 与描述子分数等权融合。
4. `shuffled_descriptor`：只打乱描述子对应帧，其他过程与融合分支相同。

四条分支复用同一候选集合、几何输出、触发门控和同一 `obj_id` 写回。只有
`geometry_descriptor > geometry_only` 且 `geometry_descriptor >
shuffled_descriptor`，才能说明时序对齐的实例描述子提供了额外的身份判别信息。
后续帧 GT 只进入 `candidate_gt_iou` 和最终 IoU 指标，不参与候选排序。

```
gate(k, t) = 1[ c_t^k > τ_track ] · 1[ σ_t^k > τ_geo ] · 1[ Δt_visible^k > τ_persist ]
```

- `c_t^k`：SAM3 的跟踪/匹配置信度
- `σ_t^k`：几何分支该实例区域的深度置信度均值
- `τ_persist`：该实例被连续跟踪的帧数阈值（防止刚出现/刚重识别的目标立刻被用来做强约束）

只有三者都满足时，该实例才能：(a) 参与相机回环约束，(b) 触发3D fallback 检索。这个设计直接回应了 SAM3-DMS 论文指出的"群体级置信度更新导致ID漂移"问题——你的门控是**实例解耦**的，且额外引入了几何置信度作为交叉验证，理论上应该比 SAM3 原生的 memory selection 更鲁棒，这也可以作为论文里一个直接对比的消融点。

---

## 5. 训练数据构建

真实数据里同时有「相机位姿GT + 多目标实例mask GT + 长视频/大视角跨度」的数据集基本不存在，建议三源混合：

1. **合成数据**（主力）：Kubric / ParallelDomain / Infinigen 之类可以直接渲染出精确相机位姿、深度、实例mask，且能人为控制视角跨度、遮挡重现频率，适合做受控消融实验。
2. **伪标签真实数据**：在 ScanNet/Replica/自采数据上跑 SfM 或者现成的位姿估计模型拿相机位姿伪GT，用 SAM3 自标注拿实例mask伪GT，构造弱监督训练对。适合做domain adaptation，防止合成数据的sim-to-real gap。
3. **对抗性回环子集**：专门构造"物体消失后大视角切回再出现"的片段（合成数据里容易控制），这是你的方法最应该体现优势的场景，也是评估的核心子集。

训练策略建议：桥接模块 + 位置编码 adapter 用合成数据从零训练，用伪标签真实数据做微调，几何/跟踪两个主干本身**冻结或只做LoRA级别微调**，避免灾难性遗忘各自原有的强能力。

---

## 6. 分阶段实验计划

| 阶段 | 目标 | 方法 | 对比对象 |
|---|---|---|---|
| A | 验证跟踪信号能否抑制相机漂移 | 只做第2节的单向约束，几何分支只读SAM3输出 | StreamVGGT / XStreamVGGT / InfiniteVGGT（纯几何baseline） |
| B | 验证几何先验能否提升大视角re-id | 只做第3节的单向增强，SAM3只读几何输出 | SAM3 / SAM3-DMS（纯跟踪baseline） |
| C | 双向耦合 + 门控 | 完整系统，重点看是否收敛、是否有正反馈震荡 | A+B简单叠加 vs 完整耦合 |
| D | 联合产出评估 | 实例级语义点云地图构建 | 无直接baseline，需自建评测集 |

强烈建议**先做A、B，再做C**，理由前面讨论过：双向耦合系统若一开始就纠缠，出问题时很难定位是哪个方向的约束在起负作用。

---

## 7. 评估协议

- **相机位姿漂移**：ATE/RPE，数据集用 ScanNet、TUM RGB-D、7-Scenes，重点看长序列（>500帧）和大视角切换片段的漂移。
- **跟踪一致性**：DAVIS17、YouTube-VOS 2019、MOSE、SA-V（SAM3 原生 benchmark），额外自建"大视角跨度+目标消失重现"子集专项评估 ID switch 率。
- **联合评估（无现成benchmark，需自建）**：在合成数据上同时有位姿GT+实例GT+点云GT，设计一个联合指标，例如"实例级点云一致性误差"（同一实例在不同帧重建出的点云配准误差）+ "ID-位姿联合漂移曲线"（把两者的误差累积画在同一时间轴上，直观看两个分支互相修正的效果）。这个自建评测集本身可以作为论文的一个附带贡献。

---

## 8. 主要风险点

1. **互相依赖的死锁风险**：几何分支漂移时输出的位姿会误导 fallback 检索，跟踪分支ID错误时会误导相机约束。门控机制是唯一的防线，需要重点做消融证明门控确实必要（去掉门控 vs 保留门控的对比实验，几乎是必须有的一组消融）。
2. **StreamVGGT 前馈模型缺少显式优化环节**，第2节的约束在推理时的落地方式需要额外设计一个轻量精修层，这部分工程量不小，建议提前预研。
3. **训练数据稀缺**，合成数据的sim-to-real gap可能是最终效果的主要瓶颈，需要预留时间做domain adaptation实验，而不是假设合成训练直接可用。
4. **效率**：两个causal大模型的KV cache都需要维护，建议从系统设计初期就引入 KV cache 压缩（可直接借鉴 XStreamVGGT 的剪枝+量化思路），否则"流式"这个卖点在长序列上可能名存实亡。

---

## 9. 建议的最小可行实验（第一周就能跑起来的版本）

在合成数据的一个短视频片段上，只做 §2.2 的静态物体先验约束（最简单、最不容易调试出问题的一条），加上 §4 的门控，先验证"这个约束方向是否为正"，再逐步加入回环残差和跟踪增强模块。这样可以最快拿到第一个可用于判断整体方向是否成立的信号。
