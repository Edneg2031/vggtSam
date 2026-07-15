# 流式几何重建 × 开放词表实例跟踪 耦合系统技术方案细化

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

当前受控实现先比较 `GT oracle / SAM3 original / hard-recovery memory` 三种
实例 mask。mask 从 StreamVGGT 的 `depth_head + camera_head` pointmap 选择同一
静态实例点，与 reference 实例点做相同的 translation-only ICP；接受后的
`Delta T` 同时更新整帧 camera pose 与整帧 pointmap。该实验不训练 backbone，
先对 translation 增量使用 `alpha={0,0.25,0.5,1}` 做单变量阻尼消融，不同时
加入 BA 或多实例约束。该实验用于区分“mask 不可靠”和“单实例位姿增量过强”。

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
