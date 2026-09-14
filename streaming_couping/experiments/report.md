# 用 SAM 物体锚点修正 HorizonStream 位姿漂移 —— 实验报告

2026-09-14 · 场景 ScanNet++ `00a231a370` · 帧窗 `90–189`（100 帧）与 `90–239`（150 帧）

---

## 1. 问题

HorizonStream 是流式的几何基础模型：给它一段 RGB，它逐帧输出 metric depth、depth
confidence 和相机位姿。位姿是在线因果累积的（`online_motion_averaging`，每个新位姿
取一个 9 窗口的中值），**误差会累积且不会自我纠正** —— 它没有任何跨帧的景物级约束。

SAM3.1 给的是另一类东西：文本 prompt 命中的物体 mask，以及每帧保持一致的
**persistent instance ID**。同一个物体在第 10 帧和第 200 帧拿到的是同一个 ID，
所以它在两帧里的位置差异**本身就蕴含一个位姿约束**。

本实验要验证的就是这件事：

> **SAM 的 persistent object tracks 能不能当跨帧稳定的物体锚点，检测并修正
> HorizonStream 的累计位姿漂移，并把修正反馈进后续帧？**

约束条件：**不训练任何模型**，不改 HorizonStream 的 backbone / KV / GLA cache，
不引入 DINO 或任何 learned score。GT 只在所有反馈决策冻结之后加载，**仅用于评测**。

---

## 2. 方法

### 2.1 总体结构

四个阶段，前两个各跑一次模型，后两个是纯 CPU：

```
Stage 1  几何   HorizonStream（冻结）
         → metric depth、depth confidence、intrinsics、在线因果位姿、chunk camera maps

Stage 2a 语义   SAM3.1（冻结）+ 每 (frame, instance) 的 6DoF 提案
         → 语义地图 + feedback_diagnostics.pt（观测点云 / 参考云 / 提案）

Stage 2b 分析   共识 → 门控 → 注入重放（纯 CPU）
         → 各变体的修正轨迹、逐帧指标、GO/NO-GO 判定

Stage 3  评测   GT mask + depth + GT 位姿 → 物体点云指标
```

**两遍式设计**：所有提案都基于 **raw 几何**计算（相机位姿不被修正污染），SAM 和
refiner 只跑一次；之后所有共识、门控、阈值迭代只重跑 Stage 2b —— 纯 CPU，几秒。

### 2.2 每一步具体做了什么

**① 提案：每个物体解一个 6DoF 增量**

对每个 (frame, instance)：mask ∩ 有效深度取 ≤256 个点，得到一个相机系点云。
拿它和该实例的**参考云**做最近点对齐，解一个 6DoF 增量 `ΔT_k`：

- 参考云 = 帧 0–4 的固定 anchor（权重 1.0）+ 近期 history（权重 0.50）；
- 最近点匹配用 mutual-NN，阈值 0.25 m，trim 70%，每对至少 8 组匹配；
- 解出的增量**左乘**在 raw pose 上：`T(δ) = [[Exp(ω), t], [0, 1]] · T_raw`；
- 幅度 clamp：≤10° / 0.25 m；
- 损失 = `‖T(δ)·P_current − Q_reference‖` 的加权 Huber（δ=0.05 m）+ `0.02·prior`。

每个实例**独立**解自己的，不共享位姿。

**② 可靠性：两个规则化分数（无 learned score）**

- `S_sem`：track 长度 / 可见率 / mask 像素 / SAM score 加权归一；
- `S_geo`：对齐改善 / inlier ratio / overlap / 修正幅度 sanity /
  **协方差特征值退化分类**（λ1≥λ2≥λ3 → volumetric / planar / linear / degenerate）。

**③ 跨物体共识**

把每个 `ΔT_k` 转成 `ξ_k = (rotvec, translation)`，做**加权 Huber IRLS**，
输出加权中位数作为共识修正 `ΔT_camera`。同时给出每个提案的 `consensus_inlier`
标志。五个消融变体一次算完（重放是 CPU）。

**④ 门控**

按优先级输出拒绝原因：`insufficient_objects → low_track_confidence →
degenerate_geometry / correction_too_large / low_geometry_confidence → no_consensus →
correction_too_large（共识幅度）→ no_alignment_improvement`。

最后一道门用**真实参考云**复算加权 NN 残差：共识修正统一作用到所有贡献物体的当前
世界点后，**残差必须严格下降** —— 不能用 per-instance 最优 loss 代替。

**⑤ 反馈**

接受则把 `ΔT_camera` 作为**绝对目标**写入 `online_absolute_poses` 与
`last_absolute_poses`（经 POC 原语 `_replace_internal_pose_for_public_target`）。

语义是**绝对目标而不是增量**：anchor（前 5 帧）钉在 raw 世界系上，所以重复注入是
覆盖，不会叠加成过修正。

### 2.3 一个决定成败的设计点

**加权方式。** `robust_semantic` 只用 `S_sem` 加权；原主方法
`robust_semantic_geometric` 额外乘了 `S_geo`。而 `S_geo` 经实测是**反预测**的 ——
边际 ρ +0.143，类内 ρ **+0.342**（正号意味着"几何置信度越高，提案越差"）。

结果：增益从 **+14.25% 直接减半到 +7.83%**，而且过不了旋转守卫。
**差别全在这一项。**

---

## 3. 实验过程

### 3.1 怎么跑

一条命令跑完整轮（含 GPU）：

```bash
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt
```

它会依次：跑 Stage 1（150 帧）→ Stage 2a 一次 → GT-mask oracle 对照（CPU）→
重放确定性门（replay 必须复现 baseline 的提案，否则整轮中止）→ 2 次 baseline 重复
（算噪声底）→ 4 个分支的 CPU 重放 + 各分支 Stage 2b → 两张对照表。

### 3.2 实验矩阵

两个维度独立变化，各自是一"代"（generation），互不覆盖：

| 维度 | 取值 |
|---|---|
| **帧窗** | `90–189`（100 帧）、`90–239`（150 帧） |
| **prompt 集** | v1 `bed wardrobe chair rug dustbin`（5）、v2 加 7 个刚性物体（12）、v3 = v1 减 `dustbin`（4） |
| **提案侧分支** | `baseline`（joint 6DoF + 永久 anchor）、`fresh`（有界参考年龄）、`factorized`（先旋转后平移）、`fresh_factor`、`translation_only` |
| **共识变体** | `single` / `mean` / `robust` / `robust_semantic`（主）/ `robust_semantic_geometric` |

帧窗和 prompt 集都定义在 `object_pose_feedback_env.zsh` 与 sweep 顶部的一处，
run 目录名由它们推出，所以两个窗口的 run 永不互相污染。

### 3.3 判据：七条，事前声明

判定**只看位姿指标，从不看 ICP loss**（loss 下降不代表位姿变好，见 §5.3）：

| # | 判据 | 阈值 | 为什么要有它 |
|---|---|---|---|
| 1 | `direct_ate_improvement_ratio` | ≥ **0.05** | 主指标 |
| 2 | `future_translation_gain_median_m` | > **0** | 修正是要**反馈到后续帧**的，只看当帧不够 |
| 3 | `future_translation_gain_positive_ratio` | ≥ **0.60** | 中位数为正可能靠少数帧拉起来 |
| 4 | `rpe_translation_ratio`（排除修正边界） | ≥ **−0.05** | 不许把逐帧相对误差改坏 |
| 5 | `accepted_ratio` | ≥ **0.10** | 只在极少数帧生效不算方法 |
| 6 | `sim3_ate_improvement_ratio` | ≥ **0** | 见下 |
| 7 | `future_rotation_gain_median_deg` | ≥ **0** | 见下 |

**第 6 条（sim3）为什么必要**：direct ATE 单看不够 —— 修正可以纯靠吸收一个全局
相似变换（scale/rigid）把它改善，而相对轨迹没变。100 帧那轮**两种失败模式都出现过**：
主变体 direct ATE +5.3% 而 sim3 −0.6%。

**第 7 条（旋转）为什么必要**：旋转估计比它要修的噪声底还差（共识旋转误差 0.61°
对 raw 逐帧 RPE rotation 0.318°）。要求它为正，是为了不让"只帮了平移"的修正
以位姿改善的名义通过。

### 3.4 测量能力（前提，不是结果）

**pipeline 是可复现的**：三次同配置重跑，`d_ATE` 散布 **1.8e-05**（100 帧）/
**4.5e-05**（150 帧），`proposal_count` 与 `accepted_frame_count` 散布 **0**。

所以下面表里百分之几量级的差异**全部有效**。唯一要小心的是 <0.01° 的旋转角，
那个尺度上测量本身有 ~1e-2 的相对漂移。

（早期曾有一个"噪声 0.026 是效应 4 倍"的说法，后来作废 —— 那是拿**跨代码版本**的
run 相比，不是重复实验。）

---

## 4. 实验结果

### 4.1 主结果：位姿

**同一个配置在两个帧窗上都通过全部七条判据：**

| 帧窗 | raw direct ATE | proposals | accepted | d_ATE | d_sim3 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 100 帧 | 0.1166 m | 229 | 33 | **+14.25%** | **+5.35%** | **GO**（7/7） |
| 150 帧 | 0.1095 m | 335 | 33 | **+15.60%** | **+4.82%** | **GO**（7/7） |

`d_sim3 > 0` 单独值得指出：增益**不是**全局相似变换的 gauge 修正，而是相对几何
本身变好了。这正是判据 6 要分开的两种情况。

通过的共识变体是 `mean` / `robust` / `robust_semantic` 三个，**恰好也是点云指标
最好的三个** —— 轨迹判据与点云指标是两个独立的测量，它们给出同一个排序。

### 4.2 主结果：改善传导到了产物

位姿指标好 ≠ 产物好，所以单独做了这个测量：把诊断里每个观测的**同一批**相机系点，
分别用 raw 轨迹和修正轨迹投到世界；参考云由 **GT mask + depth + GT 位姿**构成。
同批点、只换位姿，所以身份错误、mask 质量、采样全部抵消，变的只有"点被放到哪里"。

池化 `[all]`，100 帧，`robust_semantic` vs raw：

| 指标 | raw | 修正后 | Δ |
|---|---:|---:|---:|
| object_accuracy_m ↓ | 0.0167 | **0.0119** | **−29%** |
| object_completeness_m ↓ | 0.0947 | **0.0884** | −6.7% |
| fscore_5cm ↑ | 0.4895 | **0.5487** | **+12.1%** |
| voxel_iou_5cm ↑ | 0.1142 | 0.1144 | 持平 |
| ghost_point_ratio ↓ | 0.0926 | **0.0518** | **−44%** |

### 4.3 消融：五个提案侧分支

150 帧，噪声底 4.5e-05：

| 分支 | prop_err | cons_err | accepted | d_ATE | d_sim3 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| **baseline** | 0.0892 | 0.0786 | 33 | **+15.60%** | **+4.82%** | **GO** |
| factorized | 0.0940 | 0.0802 | 33 | +16.66% | +5.11% | GO |
| translation_only | 0.0968 | 0.0809 | 27 | +6.89% | −5.21% | NO_GO |
| fresh | 0.0891 | 0.0811 | 34 | +3.46% | −1.96% | NO_GO |
| fresh_factor | 0.0999 | 0.0851 | 29 | −0.23% | −1.00% | NO_GO |
| oracle_mask（诊断上界） | 0.1010 | 0.0936 | 22 | +0.28% | +0.04% | NO_GO |

**两点必须一起说：**

**`factorized`（先旋转后平移）在 150 帧全过，但它不稳定。** 同一个分支：
100 帧 v1 **+2.99%**（不过）→ 150 帧 v1 **+16.66%**（全过）→ 100 帧 v3 **+19.12%**
（差一条）。跨配置从 +3% 摆到 +19% 的分支**不是更好的配置，是不稳定的配置** ——
只报好的那次是误导。**能重复的是 `baseline`。**

**`oracle_mask` 是诊断对照，不是方法分支。** 它用 **GT mask 替换 SAM**（这是唯一
允许 GT 进入候选生成的地方），用来把误差拆成"分割模型错了"和"对应/几何退化是极限"。
两个窗口上它的 `prop_err` 都**比 SAM 差**（100 帧 0.1023 vs 0.0845；
150 帧 0.1010 vs 0.0892）—— **换 GT mask 不帮忙**，说明瓶颈不在 mask 质量。

### 4.4 机制验证：环路本身是通的

在 SAM 参与之前先单独验证环路：把**已知正确**的修正（来自 GT）单次写回累积器，
此后 t+1..t+10 的 translation **10/10 全部改善**。

这一条把"环路能不能接受并传播一个修正"和"能不能估出这个修正"分开了。**前者是通的。**

---

## 5. 结论

### 5.1 成立的

在冻结的 HorizonStream 上，用 SAM3.1 的 persistent object tracks 做跨物体共识、
把修正反馈进因果位姿累积器，**位姿与点云产物同时改善，两个帧窗上都通过全部七条
预先声明的判据**：

- 位姿 **+14.25% / +15.60%**（direct ATE），sim3 同为正；
- 点云 accuracy **−29%**、ghost **−44%**、F5cm **+12%**，且与轨迹判据独立地给出同一排序；
- 环路本身得到独立验证（GT 修正 10/10）；
- 结果**不依赖单一物体**：去掉 `dustbin` 后 d_ATE 不变、仍 GO。

### 5.2 适用范围（不可省）

| 项 | 状态 |
|---|---|
| 场景数 | **1** |
| 帧窗 | 2 个（100 / 150 帧），**同一场景的复核**，不是第二个场景 |
| 协议角色 | development window：判据与阈值是在这段窗口上选的，**无 held-out 证据** |
| prompt 集 | **方法的一部分，不是自由参数**（见下） |
| 点云绝对数值 | **不可引用**：预测云稀疏约 100 倍（bed 参考云 258 万点、预测约 2.5 万），completeness / recall 被稀疏性主导，**只有 raw-vs-修正的对比有效** |
| 改善的分布 | **不均匀**：`bed` 五项全改善；`chair` / `wardrobe` 是 F5cm 升但 ghost / IoU 降 |

**prompt 集为什么是方法的一部分**：把 prompt 从 5 个扩到 12 个，同一配置从
**+14.25% 翻到 −2.38%**。原因是分割器对 prompt 的响应**不可加** —— v2 里
`wardrobe` 消失了（换成了 `cabinet` 和 `window`），补进去的 7 个词有 5 个一条
proposal 都没产出。机制在代码里：每个 prompt 单独检测，但检测之后候选**跨 prompt
共享**一个接受列表，按**出生帧**排序，与已接受 track 重叠的后来者当重复丢掉。

### 5.3 明确否证的（都实测过）

| 假设 | 检验 | 结果 |
|---|---|---|
| 瓶颈在 SAM 的 mask / identity | mask oracle（GT mask 替换 SAM） | **否**，prop_err 反而更差 |
| 参考集太旧 | `fresh` 分支（anchor 年龄 37.5 → 8） | **否**，相关性 0.79 → 0.17 但 prop_err 纹丝不动 |
| 旋转/平移互相补偿 | `factorized` | **否**，100 帧上比 raw 更差 |
| 只保留平移 | `translation_only` | **否，更差**：旋转与平移在联合解里耦合，强制 ω=0 让重叠变差、平移也跟着变差 |
| 优化目标不匹配 | 换成对 GT 的 loss | **不是全部**：目标函数对**共模误差免疫** —— `P` 和 `Q` 来自同一条 depth+pose 流水线，共同误差在相减时抵消，所以 loss 看不见它，而它恰恰贡献绝对位姿误差 |

---

## 6. 复现

```bash
# 完整实验：分支 sweep + mask oracle + 噪声底（含 GPU）
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 只重读判定 + 点云评测（纯 CPU，几秒）
zsh streaming_couping/commands_check_object_pose_feedback_decision.txt

# 单元测试 + 逐类别解释 + prompt 账本（纯 CPU，只读）
zsh streaming_couping/commands_verify_object_pose_feedback.txt

# 读回上一个帧窗（默认是当前窗口）
OBJECT_POSE_FEEDBACK_FRAME_COUNT=100 zsh streaming_couping/commands_check_object_pose_feedback_decision.txt
```

配套文档：

- `experiments/results_summary.md` —— 一页数字
- `experiments/object_pose_feedback_go.md` —— 完整记录（含逐项证据与 §7 范围讨论）
- `docs/object_pose_feedback.md` —— 实现与判据定义
- `docs/object_pose_pipeline_summary.md` —— 失败路径与归因分析

---

## 附：一个未解释的观察

`anchor_rho`（anchor 年龄与 GT 修正误差的类内相关）在 100 帧是 **0.79**，到 150 帧
掉到 **0.09**。它在 100 帧上是最强的预测因子（整个 `fresh` 分支就是为它做的），
窗口变长、anchor 更老（37.5 → 52.5 帧）之后理应保持，不该消失。**目前没有解释，
所以不写进结论。**
