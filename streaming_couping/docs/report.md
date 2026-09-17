# 技术报告：SAM 物体锚点修正 HorizonStream 位姿漂移

2026-09-17

本文档分两部分。**方法**（§1–§9）说明 pipeline 如何搭建、每一步做什么、判据是什么；**实验**（§10–§20）记录结果、消融与已排除的方向。交接事项见 [`../../HANDOVER.md`](../../HANDOVER.md)。

> 本文合并自原 `method.md`、`experiments.md` 及更早的 13 份文档，原文见 git 历史。

**场景** ScanNet++ `00a231a370`；**帧窗** `90–189`（100 帧）与 `90–239`（150 帧）。

---

## 1 一段话

在冻结的 HorizonStream 上逐帧拿到**相机系** metric depth 和**因果累积**的位姿；同时用冻结的 SAM3.1 按文本 prompt 出 mask 和 persistent instance ID。对每个 (帧, 实例)，取 mask ∩ 有效深度（confidence ≥ 0.3）最多 **256 个相机系点**，与该实例的**参考云**（前 5 帧 anchor 权重 1.0 + 近期 history 权重 0.5）做 mutual-NN 匹配（0.25 m、trim 70%、每对 ≥8 组），再用 **ICP 式交替**解一个 6DoF 增量 —— 4 次外层迭代，每次重新匹配后用 Adam 跑 30 步，优化 `‖T(δ)·P − Q‖` 的加权 Huber 残差，`δ` 以 `T(δ)·T_raw` **左乘**在 raw 位姿上，clamp 到 10° / 0.25 m。把各物体的 `log(ΔT)` 做加权 Huber IRLS 取**加权中位数**（权重只用 `S_sem` —— 这是成败关键，多乘 `S_geo` 会让增益减半），过门控，接受的修正作为**绝对目标**写回 `online_absolute_poses` / `last_absolute_poses`。

整体是**两遍式**：提案全部基于 raw 几何一次算完，修正之后只在 CPU 上因果重放累积器 —— 所以**修正在传播上是闭环的、在估计上是开环的**。

不训练；不改 HorizonStream backbone / KV / GLA cache；GT 只在所有决策冻结后加载，仅用于评测。

---

## 2 系统结构

### 2.1 几何：HorizonStream（冻结）

流式 chunk 模型。输入一段 RGB，逐帧输出 metric depth、depth confidence、intrinsics。

**相机位姿不是模型直接输出的绝对轨迹**，而是运行时累积出来的：

| 层 | 内容 |
|---|---|
| 模型 | 按 chunk（`window_size=10`，`sliding=1`）输出 chunk 内的**相对位姿**（w2c，锚定窗口最新帧） |
| 运行时 | `online_motion_averaging` 把相对位姿**累积**成绝对轨迹 |
| 抗噪 | 相邻 chunk 重叠，同一新帧可从多个锚点链回去 → **9 个候选绝对位姿** → 取**旋转中值 + 位置中值** |
| 表示 | 内部是 **w2c**；公开轨迹是 **c2w、frame-0 gauge** |

**两条并行的传播路径**：`online_absolute_poses`（中值路径）与 `last_absolute_poses`（只走上一帧的相对位姿）。注入修正必须**两个都写**，否则只沿一条路传播。

### 2.2 语义：SAM3.1（冻结）

文本 prompt → mask + **persistent instance ID**（同一物体跨帧同一个 ID）。`track_all_forward(image_paths, prompt)` 只吃 **RGB**，和位姿无关。

### 2.3 四阶段

```
Stage 1  几何   HorizonStream → depth / confidence / intrinsics / 因果位姿 / chunk camera maps
Stage 2a 语义   SAM3.1 + 每 (frame, instance) 的 6DoF 提案 → 语义地图 + feedback_diagnostics.pt
Stage 2b 分析   共识 → 门控 → 注入重放（纯 CPU）→ 各变体轨迹 + 逐帧指标 + GO/NO-GO
Stage 3  评测   GT mask + depth + GT 位姿 → 物体点云指标
```

**两遍式设计的意义**：SAM 与 refiner 只跑一次，之后所有共识 / 门控 / 阈值迭代**只重跑 Stage 2b（纯 CPU，几秒）**。代价是提案看不到修正（§6.1）。

---

## 3 一步步做什么

### ① 提案：每个物体解一个 6DoF 增量

对每个 (帧, 实例)：mask ∩ 有效深度取 ≤256 点，与该实例的**参考云**做最近点对齐。

- **参考云** = 帧 0–4 的固定 anchor（权重 1.0）+ 近期 history（权重 0.50）
- 匹配：mutual-NN，阈值 0.25 m，trim 70%，每对 ≥8 组匹配
- 求解：`δ = (ω, t)`，位姿 `= [[Exp(ω), t], [0,1]] @ T_raw`（**左乘**）
- **ICP 式交替**：4 次外层迭代，每次**重新匹配**后跑 30 步 Adam（lr 0.03，Huber δ 0.05 m）
- 损失：加权 Huber 点到点残差 + `0.02 × prior`
- clamp：≤10° / 0.25 m

**每个实例独立解自己的**，不共享位姿。

### ② 可靠性：两个规则化分数（无 learned score）

**`S_sem`**（`semantic_reliability`）= 四项加权和：

```
track_length_score   归一化到 [min_track_length, target_track_length]
visibility_score     可见率 / min_visibility_ratio
mask_score           mask 像素 / target_mask_pixels
score_component      (SAM score − 0.5) / 0.5
```

硬拒（→ `low_track_confidence`）：track 太短 / 可见率太低 / SAM score 太低。

**`S_geo`**（`geometric_reliability`）= 三项 + 一个退化阻尼：

```
improvement_score    对齐损失改善 / target_relative_improvement
inlier_score         inlier ratio / target_inlier_ratio
overlap_score        重叠点数 / target_overlap_points
degeneracy_damp      协方差特征值比（volumetric ←→ degenerate），只降权不直接拒
```

硬拒：`degenerate_geometry` / `correction_too_large` / `low_geometry_confidence`。

### ③ 跨物体共识

把每个 `ξ_k = log(ΔT_k)`（rotvec + translation，与优化变量同参数化）做**加权 Huber IRLS**，输出**加权中位数**作为 `ΔT_camera`。每个提案被标上 `consensus_inlier`。

五个消融变体一次算完（重放是 CPU）：

| 变体 | 权重 |
|---|---|
| `single` | 最高权重单物体，`S_sem × S_geo` |
| `mean` | 均匀，无 IRLS |
| `robust` | 均匀 + IRLS |
| **`robust_semantic`（主方法）** | **只 `S_sem`** |
| `robust_semantic_geometric` | `S_sem × S_geo`（实测有害） |

### ④ 门控

按优先级输出拒绝原因：

```
insufficient_objects（<2 个可靠物体）
  → low_track_confidence → degenerate_geometry / correction_too_large / low_geometry_confidence
  → no_consensus → correction_too_large（共识幅度）
  → no_alignment_improvement
```

最后一道用**真实参考云**复算加权 NN 残差：共识修正统一作用到所有贡献物体的当前世界点后，**残差必须严格下降** —— 不能用 per-instance 最优 loss 代替。

### ⑤ 反馈

接受的修正作为**绝对目标**写入 `online_absolute_poses` / `last_absolute_poses`（经 `_replace_internal_pose_for_public_target`，反解内部该存什么）。

**绝对目标而非增量**：anchor（前 5 帧）钉在 raw 世界系上，所以重复注入是**覆盖**，不会叠加成过修正。

---

## 4 位姿约定

| 对象 | 约定 |
|---|---|
| 模型 chunk 输出 / 内部累加器 | w2c，窗口锚定最新帧 |
| 公开轨迹 / 评测 | c2w，frame-0 gauge |
| 物体提案 ΔT | c2w 左乘：`C[k,t] = T_aligned[k,t] @ inv(T_raw[t])` |
| GT 修正 | `ΔT_GT_t = G_t @ inv(R_t)`（同一 gauge） |
| 注入 | 绝对目标 `target_t = ΔT_consensus @ R_t` |
| GT 来源 | manifest `world_to_camera` 求逆 → 归一到首个选中帧；**仅在所有反馈决策冻结之后加载** |

---

## 5 物体筛选：六层

修正质量取决于**哪些物体参与了投票**，而物体集合在**六层**被改变。**关键是 ① 在 ②–⑥ 之前** —— ① 出局的物体，后面五层从没见过它。

```
① 跟踪层   哪些物体**存在**？  词表 · 全局 16 名额 · 跨 prompt 判重（按出生帧）
② 观测层   能否**形成提案**？  track score 0.50 · static 0.20 · mask ≥32px · 面积 ≤0.85
                              · 几何置信度 0.30 · ≥24 点 · 每对 ≥8 匹配 · 总 ≥16
③ 可靠层   算不算**可信**？    S_sem · S_geo（§3②）
④ 共识层   进不进**中位数**？  Huber IRLS → consensus_inlier
⑤ 门控层   这一帧**用不用**？  物体数 · 集中度 · 幅度上限 · 对齐损失必须下降
⑥ 加权层   各占**多少权重**？  robust_semantic = 只按 S_sem
```

### ① 的细节，因为它最要紧

| 判据 | 取值 |
|---|---|
| `max_objects_per_prompt` | 16（**每个词**最多回几条） |
| **`--max-objects`（全局）** | **16（所有词加起来）** |
| 判重 `duplicate_iou` | 0.80 |
| `min_birth_pixels` | 128 |

**这一层没有任何质量判据**：排序键是 `(出生帧, prompt 序号, obj_id)`，名额满了就 `break`，后面的候选**连判重都不会被检查**。

**实测（v5：v1 + `cabinet`）**：`cabinet` 一个词回了 15 条 track，accepted 合计**正好撞满 16**，`rug` 被挤掉一条，而 `wardrobe` **被判重丢掉**（输给出生更早的 track），结果从 +15.36% 掉到 **−11.87%**。

### 各层判据实测有没有用

类内 ρ = 固定类别后的 Spearman；保留 vs 丢弃 = 该筛选留下的提案与丢掉的**中位 GT 误差**。

| 判据 | 数字 | 结论 |
|---|---:|---|
| `S_sem`（`semantic_confidence` 代理） | 类内 ρ = **+0.634** | **反预测** —— 分数越高，提案越差 |
| `S_geo`（`geometry_confidence` 代理） | 类内 ρ = **+0.342** | **反预测** |
| `reliable` 硬拒筛选 | 保留 0.0881 vs 丢弃 0.0810 | **零区分度** |
| **`consensus_inlier`** | 保留 **0.0753** vs 丢弃 **0.1573** | **有效（差 2 倍）** |
| **帧级 gate** | 接受 **0.0405** vs 拒绝 **0.0914** | **有效（差 2.3 倍）** |
| `track_length` | 类内 ρ = **+0.78** | 最强预测因子，但方向是**负的**（track 越老越差），且**事后才有** |

**两个设计来做"物体质量"判断的分数是反的；两个真正有效的都是"一致性"判据。**

于是判据按**生效时间**分成三类，而它们并不在同一个位置：

| 类别 | 例子 | 什么时候能算 | 效果 |
|---|---|---|---|
| **物体自身质量** | `S_sem`、`S_geo`、`reliable` | 拿到这个物体就能算 | **反预测 / 零区分度** |
| **跨物体一致性** | `consensus_inlier`、帧级 gate | **必须已经有一群物体在投票** | 有效 |
| **出生顺序** | 名额、判重 | 跟踪时 | 与质量无关，但**决定谁在场** |

**所以要控制"哪些物体参与修正"，能改的是 ① 层的规则**（名额、判重排序、按输入侧测量挑选 prompt 集），**不是加一个更好的打分器** —— ① 出局的物体，后面五层从没见过它。

---

## 6 两处必须分清的机制

### 6.1 环路闭在哪一半

**传播上闭环、估计上开环。** 修正会经累积器带到后面每一帧；但**提案是在任何修正存在之前一次算完的**，所以第 t 帧的修正改变不了第 t+1 帧怎么估计。

能闭环的只有**参考云**这一半 —— 因为 SAM 只看 RGB、模型 `forward_chunk` **不含位姿**、depth 是**相机系**的：

| 量 | 修正位姿会改变它吗 |
|---|---|
| SAM 的 mask | **不** |
| depth | **不** |
| **参考云**（靠位姿进世界） | **会** ← 唯一 |

（这条已经实测过：闭环残差降 18% 但轨迹差 3.6%，见 §16.1。）

### 6.2 "修点云"有两个

| | (a) 逐实例修点云 | (b) 位姿反馈（**主线**） |
|---|---|---|
| 相机位姿 | **保持 raw** | **被修正** |
| 改的是什么 | 每个实例各修自己的点云 | **放置点的那个位姿** |
| 分支名 | `object_pose_object_only_per_instance` | object-consensus feedback |
| 修正量 | `storage_pose @ inv(raw_pose)`，只作用于该实例的点 | 全部物体投票出的每帧共识 |

**(b) 里点云一个点都没改** —— 同一批相机系点，被不同的轨迹摆到不同的世界位置。拿 (a) 的逐实例好坏去推断它在 (b) 的共识里话语权多大是错的。

---

## 7 判据：七条，事前声明

判定**只看位姿指标，从不看 ICP loss**：

| # | 判据 | 阈值 | 为什么 |
|---|---|---|---|
| 1 | `direct_ate_improvement_ratio` | ≥ 0.05 | 主指标 |
| 2 | `future_translation_gain_median_m` | > 0 | 修正是要**反馈到后续帧**的 |
| 3 | `future_translation_gain_positive_ratio` | ≥ 0.60 | 中位数可能靠少数帧拉起 |
| 4 | `rpe_translation_ratio`（排除修正边界） | ≥ −0.05 | 不许把逐帧相对误差改坏 |
| 5 | `accepted_ratio` | ≥ 0.10 | 只在极少数帧生效不算方法 |
| 6 | `sim3_ate_improvement_ratio` | ≥ 0 | 见下 |
| 7 | `future_rotation_gain_median_deg` | ≥ 0 | 见下 |

**第 6 条（sim3）为什么必要**：direct ATE 单看不够 —— 修正可以纯靠吸收一个全局相似变换（scale/rigid）把它改善，而相对轨迹没变。100 帧那轮**两种失败模式都出现过**。

**第 7 条（旋转）为什么必要**：旋转估计比它要修的噪声底还差（共识旋转误差 0.611° 对 raw 逐帧 RPE rotation 0.318°）。要求它为正，是为了不让"只帮了平移"的修正蒙混过关。

---

## 8 两个必须小心的测量前提

### 8.1 噪声底有两种，别用一个数当全部

| 误差来源 | 数字 | 怎么测的 |
|---|---|---|
| 分割 + 分析链（**几何固定**） | **1.8e-05** | 三次重跑，`--reuse-if-valid` 复用同一份 stage-1 缓存 |
| **独立 stage-1 重跑** | **≈ 2.4e-03** | 两次完整重跑（清空 `outputs/` 后重建） |

**第二种才是"别人重跑你的实验"会遇到的差**，比第一种大约 **130 倍**（HorizonStream 在 GPU 上不是逐位确定的）。

### 8.2 小于 0.01° 的旋转角不可信

那个尺度上测量本身有 ~1e-2 的相对漂移。旋转角用 `atan2(sin, cos)` 而非 `acos(trace)` —— 接近单位阵时迹是平的，`acos` 会把浮点噪声放大成零点几度。

---

## 9 帧窗、代数与命名

- **帧窗**与 **run 目录名**都在 `object_pose_feedback_env.zsh` 里定义一次，所有命令文件 source 它。两个窗口的 run 永不互相污染。
- **代数（generation）**是 prompt 集：`GENERATION` 一行切换，词表从 `GENERATION_PROMPTS` 按标签查出（**标签和词表不可能不一致**）。
- **帧选择会静默截断**（`positions[:count]`），所以 `opf_require_frame_window` 在任何东西碰到 GPU 之前就拒绝"场景不够长"的跑法。

---

## 10 结论提要

在冻结的 HorizonStream 上，以 SAM3.1 的 persistent object tracks 做跨物体共识并将修正反馈进因果位姿累积器，**位姿与点云产物同时改善，两个帧窗均通过全部七条事前判据**。

结论适用范围有限：单一场景、单一开发窗口、阈值即在该窗口上确定、无 held-out 证据（§17）。

---

## 11 位姿主结果

同一配置在两个帧窗上的结果：

| 帧窗 | raw direct ATE | proposals | accepted | d_ATE | d_sim3 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 100 帧 | 0.1166 m | 229 | 33 | **+14.25%** | **+5.35%** | **GO**（7/7） |
| 150 帧 | 0.1095 m | 335 | 33 | **+15.60%** | **+4.82%** | **GO**（7/7） |

150 帧一行取自**首次运行**。清空 `outputs/` 后的**独立重跑**给出 **+15.36% / +4.29%**（accepted 34），两者相差 **0.24 个百分点**，即 §8.1 所述跨 stage-1 运行的误差。本文引用 150 帧数值处均注明取自哪一次；多数场合采用重跑值。

**关于 `d_sim3 > 0`**：该结果说明增益**不是**全局相似变换的 gauge 修正，而是相对几何本身改善。这正是判据 6 用于区分的情形。

---

## 12 位姿改善向产物传导

位姿指标改善不等同于产物改善，因此单独测量：将诊断中每个观测的**同一批**相机系点，分别以 raw 轨迹与修正轨迹置入世界；参考云由 **GT mask + depth + GT 位姿**构成。点集与位姿来源分离，故身份错误、mask 质量与采样因素全部抵消。

池化 `[all]`，100 帧，`robust_semantic` 相对 raw：

| 指标 | raw | 修正后 | Δ |
|---|---:|---:|---:|
| object_accuracy_m ↓ | 0.0167 | **0.0119** | **−29%** |
| object_completeness_m ↓ | 0.0947 | **0.0884** | −6.7% |
| fscore_5cm ↑ | 0.4895 | **0.5487** | **+12.1%** |
| voxel_iou_5cm ↑ | 0.1142 | 0.1144 | 持平 |
| ghost_point_ratio ↓ | 0.0926 | **0.0518** | **−44%** |

**通过轨迹判据的三个变体，同时是点云指标最优的三个** —— 两项独立测量给出一致排序。

**类别间分布不均匀**：`bed` 五项全部改善（accuracy −37%、F5cm +18%、IoU +22%、ghost −50%）；`chair` 与 `wardrobe` 为 F5cm 上升而 ghost / IoU 下降。

---

## 13 环路机制验证

在引入 SAM 之前先行验证环路本身：将**已知正确**的修正（取自 GT）单次写回累积器，此后 t+1..t+10 的 translation 位移 **10/10 全部改善**。

该结果将"环路能否接受并传播一个修正"与"能否估出该修正"分离。**前者成立。**

---

## 14 对单一物体的依赖性

v3 = v1 移除 `dustbin`（GT 侧评分最优类别），其余配置不变：

| | v1 | v3 |
|---|---:|---:|
| 100 帧 d_ATE | +14.25% | +14.25% |
| 150 帧 d_ATE | +15.36% | +15.35% |

**两个帧窗均已验证**：移除一个词后 d_ATE 不变且仍为 GO。因此 v1 的结果并非依赖某一物体。

> 该结论不意味着任意词均可移除。v5 加入 `cabinet` 时 `wardrobe` 被挤出，结果显著劣化（§16.2）。

---

## 15 消融

### 15.1 共识侧（5 个变体，Stage 2b）

同一 run 内，各变体相对自身 raw：

| 变体 | 权重 | direct ATE Δ | 判定 |
|---|---|---:|---|
| `mean` | 均匀，无 IRLS | — | **GO** |
| `robust` | 均匀 + IRLS | **+13.94%** | **GO** |
| **`robust_semantic`** | **仅 `S_sem`** | **+14.25%** | **GO（7/7）** |
| `robust_semantic_geometric` | `S_sem × S_geo`（原主方法） | +7.83% | NO_GO |
| `single` | 单物体参考 | — | NO_GO |

**加权方式决定结果**：`S_geo` 经实测为**反预测**，叠加该项使增益由 +14.25% 减半至 +7.83%。

### 15.2 提案侧（5 个分支，Stage 2a）

**概念界定。** 本节的分支均作用于 Stage 2a 的 refiner，改变**每个物体的 `ΔT` 如何求得**，即发生在共识之前。§15.1 的共识侧变体则改变 `ΔT` 求得**之后如何合并**。

**术语。** "参考云"指求解某物体 6DoF 时，用于对齐的**该物体在更早帧的观测点**。默认由两部分构成：第 0–4 帧（运行起始时取得、此后冻结，称 anchor）+ 最近若干帧。

**各分支所检验的假设。** 单个提案的中位误差为 0.089 m，而待修正的 raw ATE 为 0.109 m，提案质量即瓶颈所在。因此每个分支检验的都是"该误差来源为何"：

| 分支 | 所检验的假设 | 检验方式 |
|---|---|---|
| `baseline` | —（**非消融**，即方法本身） | 联合求解 6DoF |
| `fresh` | **参考云过于陈旧** —— 运行至第 150 帧时 anchor 仍为第 0–4 帧。归因显示**参考越旧、提案越差**（类内 ρ = **0.79**，为各特征中最强） | 禁用 15 帧以外的参考；anchor 每 20 帧重取 |
| `factorized` | **旋转与平移在联合求解中相互干扰** —— 旋转本身为较差的一半 | 先解旋转、再解平移 |
| `translation_only` | **旋转估计质量过低，不如不予修正** | 强制 ω=0 |
| `fresh_factor` | 上述两项假设是否互补 | 两者同时启用 |
| `oracle_mask` | **瓶颈在于分割器的 mask** | 以 **GT mask** 替换 SAM |

150 帧，首次运行（噪声底 4.5e-05）：

| 分支 | prop_err | cons_err | accepted | d_ATE | d_sim3 | 判定 | **假设检验结果** |
|---|---:|---:|---:|---:|---:|---|---|
| **baseline** | 0.0892 | 0.0786 | 33 | **+15.60%** | **+4.82%** | **GO** | 报告的主结果 |
| `factorized` | 0.0940 | 0.0802 | 33 | +16.66% | +5.11% | GO | 数值最高但**不可用**：同分支跨配置摆动于 +3% 与 +19% 之间（§16 已排除方向表） |
| `translation_only` | 0.0968 | 0.0809 | 27 | +6.89% | **−5.21%** | NO_GO | **更差。** 旋转与平移在联合求解中**耦合**，去除旋转使重叠恶化，平移随之劣化 |
| `fresh` | 0.0891 | 0.0811 | 34 | +3.46% | −1.96% | NO_GO | **假设被否，且证明该相关为混淆所致。** anchor 年龄 37.5→8、ρ 0.79→0.17，而 `prop_err` **无变化**（0.0893→0.0891）。anchor 陈旧 ⟺ 处于序列后段，而序列后段本身即为漂移最大处 |
| `fresh_factor` | 0.0999 | 0.0851 | 29 | −0.23% | −1.00% | NO_GO | 两项同时启用为**最差**，证实二者不互补 |
| `oracle_mask` | 0.1010 | 0.0936 | 22 | +0.28% | +0.04% | NO_GO | **诊断上界，非方法分支。** 换用 GT mask 后 `prop_err` **反而更差**，表明**瓶颈不在 mask** |

**本表的关键列为 `prop_err`**（提案侧的直接读数）。四个消融分支中，有两个改变了提案却未曾改善提案；即"调整提案侧旋钮"未找到优于 baseline 的解。

---

## 16 已排除的技术方向

以下方向均已实测，不建议重复尝试。

| 方向 | 结论 |
|---|---|
| 改进分割质量 | `oracle_mask`（GT mask 替换 SAM）的 `prop_err` **反而更差**（0.1010 对 0.0892） |
| 更新参考集 | `fresh`：年龄 37.5→8、ρ 0.79→0.17，`prop_err` **无变化** |
| 旋转与平移解耦 | `factorized` 可全部通过，但**跨配置摆动于 +3% 与 +19% 之间**：100 帧 v1 **+2.99%**（未通过）→ 150 帧 v1 **+18.64%**（GO）→ 100 帧 v3 **+19.12%**（差一条）→ 150 帧 v3 **NO_GO**。属**不稳定配置**，非更优配置 |
| 仅修正平移 | `translation_only` **更差**，sim3 转负 |
| 扩充 prompt | v4 加入 `table`：SAM 侧**未返回任何 track**（而 GT oracle 为该词产出约 **99 条提案** —— 同一词、同一场景、同一份几何）。v5 加入 `cabinet`：返回 15 条、占满 16 名额、将 `wardrobe` 判重挤出，结果 **−11.87%** |
| 闭环 / 流式 | 见 §16.1 |
| 更换优化目标 | 目标函数对**共模误差免疫**：`P` 与 `Q` 出自同一条 depth+pose 流水线，共同误差在相减时抵消，故 loss 无法感知其贡献的绝对位姿误差 |

### 16.1 闭环（流式）方向

**定义。** "闭环"指第 t 帧的修正改变第 t+1 帧**如何估计**。当前不成立 —— 所有提案在任何修正存在之前一次算完。（修正在**传播**上是因果的，开环的是**估计**。）

**实验。** 以第一轮的修正轨迹作为新基座，用**同一批相机系点**重解提案，随后走与第一轮 **完全相同**的 Stage 2b 路径。全部为 CPU。

| | 提案残差中位（平移） | 轨迹 d_ATE |
|---|---:|---:|
| **闭环**（回灌修正） | 0.0893 → **0.0732**（**−18.0%**） | 基座 0.092648 → **0.096010**（**+3.6%**，劣化） |
| **空对照**（基座不变） | 0.0893 → 0.0894（**+0.1%**） | — |

空对照为必需项：重解路径由诊断**重建**观测与配置，自身存在偏差。实测仅 +0.1%，故 −18% 可归因于闭环本身。

**结果：残差下降 18% 而轨迹劣化 3.6%，五个变体全部 NO_GO。** 环路确实携带了第一轮未使用的信息，但该信息**未转化为更优位姿，反而使其劣化**，与"优化目标下降不等同于指标改善"一致。

劣化机制（门控所选修正更差，或修正之间相互冲突）**本轮数据无法区分**，故不作推断。

结合原理性上限（仅参考云一侧可变，见 §6.1），结论为：**流式方向不值得实施。**

### 16.2 扩充 prompt 的作用机制

v5 的候选账本首次直接观测到该机制：

```
prompt    accepted  duplicate  over_object_cap
bed       1         0          0
wardrobe  0         1          0
chair     3         0          0
rug       1         0          1
dustbin   5         0          0
cabinet   6         0          9
```

- **判重**：`wardrobe` 输给一条出生更早的 track
- **名额**：`cabinet` 返回 15 条，accepted 合计 **1+0+3+1+5+6 = 16**，恰好达到全局上限

**归因限制。** 丢失 `wardrobe` 与引入 `cabinet` 同时发生，本轮**无法区分**二者对结果的作用。分离需构造"新词进入而原词不丢失"的条件（提高 `--max-objects` 或调整判重排序）。

---

## 17 适用范围

| 项 | 状态 |
|---|---|
| 场景数 | **1** |
| 帧窗 | 2 个（100 / 150 帧），为**同一场景的复核**，非第二个场景 |
| 协议角色 | 开发窗口；判据与阈值在该窗口上确定，**无 held-out 证据** |
| prompt 集 | **方法的一部分，非自由参数**（§16.2） |
| 点云绝对数值 | **不可引用**。预测云为 refiner 采样的 ≤256 点/观测（bed 参考云 258 万点、预测约 2.5 万，稀疏约 100 倍），仅 raw 与修正的**对比**有效 |
| 改善的分布 | **不均匀**（§12） |

---

## 18 早期实验

主结果确定之前所走的路径，结论列此以免重复：

| 实验 | 结果 |
|---|---|
| V1 object point alignment | **实际未应用修正**（旋转 0.0），raw 与 aligned 指标完全相同 |
| V3 instance point alignment | 内部 RMSE 0.0369 → 0.0108（**−58%**），而 GT 指标几乎不变（F5cm +0.0004、IoU −0.0013）—— **loss 下降不等同于产物改善** |
| shared object-only 6DoF（100 帧） | `bed` 四项全部改善；但**聚合 accuracy +0.0030、ghost +0.0149，劣化** |
| per-instance object-only（100 帧） | `dustbin` track2 三项改善或持平；同批 track3 dustbin F5cm **−0.20**、`wardrobe` −0.065。**聚合 F5cm −0.028** |
| per-instance（300 帧连续） | **全部劣化**。accepted 266 帧，仅余 2 个 matched object |
| online object pose loop（100 帧） | ATE 0.1192 → 0.1244、地图 F5cm 0.348 → 0.338，**全部劣化** |
| **GT feedback POC（50 帧）** | **机制验证成功**：已知正确的修正写回累积器后，t+1..t+10 translation **10/10 改善** |

**单物体效应可复现**：`dustbin` 实例 2 的 F5cm 提升在三次独立运行中均为 **+0.086**（0.763275 / 0.763275 / 0.763452）；实例 8 稳定为 **−0.20**。同一物体会重复地获益或受损。

---

## 19 尚未解释的观察

`anchor_rho`（anchor 年龄与 GT 修正误差的**类内**相关系数）在 100 帧为 **0.79**，至 150 帧降至 **0.09**。

该量在 100 帧上为最强预测因子（`fresh` 分支即为此设计），窗口延长、anchor 更陈旧（37.5 → 52.5 帧）后理应保持。**目前无解释，故不纳入结论。**

---

## 20 复现

```bash
# 运行（GPU，唯一入口）：判定、对照表、两张图、候选账本摘要一次产出
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 读取历史帧窗（默认使用当前窗口）
OBJECT_POSE_FEEDBACK_FRAME_COUNT=100 zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt
```

**每轮运行产出两张图**（CPU，数秒），位于 `<run>.baseline/object_pose_feedback/`：

| 图 | 内容 |
|---|---|
| `pose_comparison.png` | 轨迹俯视与逐帧平移/旋转误差，三条曲线为 GT / raw / 修正后，RMSE 标注于标题 |
| `object_cloud_comparison.png` | 逐物体点云，**同一批点以三条轨迹分别置入世界**，仅位姿不同 |

本文涉及但主链路不直接产出的分析（逐类别对照、闭环实验、归因分析），其调用方式见 [`../../HANDOVER.md`](../../HANDOVER.md) §3.2。
