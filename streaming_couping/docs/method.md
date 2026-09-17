# 方法：SAM 物体锚点修正 HorizonStream 位姿漂移

2026-09-16 · 本文是**方法文档**：pipeline 怎么搭、每一步做什么、判据是什么。
实验结果与失败分析见 [`../experiments/experiments.md`](../experiments/experiments.md)；
交接须知见 [`../../HANDOVER.md`](../../HANDOVER.md)。

> 本文合并自原 `current_pipeline.md`、`system_pipeline.md`、`object_pose_feedback.md`、
> `object_pose_refinement.md`、`object_selection_criteria.md`、
> `horizonstream_gt_feedback_poc.md` 六份文档。原始文本在 git 历史里。

---

## 1. 一段话

在冻结的 HorizonStream 上逐帧拿到**相机系** metric depth 和**因果累积**的位姿；同时用冻结的
SAM3.1 按文本 prompt 出 mask 和 persistent instance ID。对每个 (帧, 实例)，取 mask ∩ 有效
深度（confidence ≥ 0.3）最多 **256 个相机系点**，与该实例的**参考云**（前 5 帧 anchor 权重
1.0 + 近期 history 权重 0.5）做 mutual-NN 匹配（0.25 m、trim 70%、每对 ≥8 组），再用 **ICP 式
交替**解一个 6DoF 增量 —— 4 次外层迭代，每次重新匹配后用 Adam 跑 30 步，优化
`‖T(δ)·P − Q‖` 的加权 Huber 残差，`δ` 以 `T(δ)·T_raw` **左乘**在 raw 位姿上，clamp 到
10° / 0.25 m。把各物体的 `log(ΔT)` 做加权 Huber IRLS 取**加权中位数**（权重只用 `S_sem` ——
这是成败关键，多乘 `S_geo` 会让增益减半），过门控，接受的修正作为**绝对目标**写回
`online_absolute_poses` / `last_absolute_poses`。

整体是**两遍式**：提案全部基于 raw 几何一次算完，修正之后只在 CPU 上因果重放累积器 ——
所以**修正在传播上是闭环的、在估计上是开环的**。

不训练；不改 HorizonStream backbone / KV / GLA cache；GT 只在所有决策冻结后加载，仅用于评测。

---

## 2. 系统结构

### 2.1 几何：HorizonStream（冻结）

流式 chunk 模型。输入一段 RGB，逐帧输出 metric depth、depth confidence、intrinsics。

**相机位姿不是模型直接输出的绝对轨迹**，而是运行时累积出来的：

| 层 | 内容 |
|---|---|
| 模型 | 按 chunk（`window_size=10`，`sliding=1`）输出 chunk 内的**相对位姿**（w2c，锚定窗口最新帧） |
| 运行时 | `online_motion_averaging` 把相对位姿**累积**成绝对轨迹 |
| 抗噪 | 相邻 chunk 重叠，同一新帧可从多个锚点链回去 → **9 个候选绝对位姿** → 取**旋转中值 + 位置中值** |
| 表示 | 内部是 **w2c**；公开轨迹是 **c2w、frame-0 gauge** |

**两条并行的传播路径**：`online_absolute_poses`（中值路径）与 `last_absolute_poses`
（只走上一帧的相对位姿）。注入修正必须**两个都写**，否则只沿一条路传播。

### 2.2 语义：SAM3.1（冻结）

文本 prompt → mask + **persistent instance ID**（同一物体跨帧同一个 ID）。
`track_all_forward(image_paths, prompt)` 只吃 **RGB**，和位姿无关。

### 2.3 四阶段

```
Stage 1  几何   HorizonStream → depth / confidence / intrinsics / 因果位姿 / chunk camera maps
Stage 2a 语义   SAM3.1 + 每 (frame, instance) 的 6DoF 提案 → 语义地图 + feedback_diagnostics.pt
Stage 2b 分析   共识 → 门控 → 注入重放（纯 CPU）→ 各变体轨迹 + 逐帧指标 + GO/NO-GO
Stage 3  评测   GT mask + depth + GT 位姿 → 物体点云指标
```

**两遍式设计的意义**：SAM 与 refiner 只跑一次，之后所有共识 / 门控 / 阈值迭代**只重跑
Stage 2b（纯 CPU，几秒）**。代价是提案看不到修正（§6.1）。

---

## 3. 一步步做什么

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

把每个 `ξ_k = log(ΔT_k)`（rotvec + translation，与优化变量同参数化）做**加权 Huber IRLS**，
输出**加权中位数**作为 `ΔT_camera`。每个提案被标上 `consensus_inlier`。

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

最后一道用**真实参考云**复算加权 NN 残差：共识修正统一作用到所有贡献物体的当前世界点后，
**残差必须严格下降** —— 不能用 per-instance 最优 loss 代替。

### ⑤ 反馈

接受的修正作为**绝对目标**写入 `online_absolute_poses` / `last_absolute_poses`
（经 `_replace_internal_pose_for_public_target`，反解内部该存什么）。

**绝对目标而非增量**：anchor（前 5 帧）钉在 raw 世界系上，所以重复注入是**覆盖**，不会叠加成过修正。

---

## 4. 位姿约定

| 对象 | 约定 |
|---|---|
| 模型 chunk 输出 / 内部累加器 | w2c，窗口锚定最新帧 |
| 公开轨迹 / 评测 | c2w，frame-0 gauge |
| 物体提案 ΔT | c2w 左乘：`C[k,t] = T_aligned[k,t] @ inv(T_raw[t])` |
| GT 修正 | `ΔT_GT_t = G_t @ inv(R_t)`（同一 gauge） |
| 注入 | 绝对目标 `target_t = ΔT_consensus @ R_t` |
| GT 来源 | manifest `world_to_camera` 求逆 → 归一到首个选中帧；**仅在所有反馈决策冻结之后加载** |

---

## 5. 物体筛选：六层

修正质量取决于**哪些物体参与了投票**，而物体集合在**六层**被改变。**关键是 ① 在 ②–⑥ 之前**
—— ① 出局的物体，后面五层从没见过它。

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

**这一层没有任何质量判据**：排序键是 `(出生帧, prompt 序号, obj_id)`，名额满了就 `break`，
后面的候选**连判重都不会被检查**。

**实测（v5：v1 + `cabinet`）**：`cabinet` 一个词回了 15 条 track，accepted 合计**正好撞满 16**，
`rug` 被挤掉一条，而 `wardrobe` **被判重丢掉**（输给出生更早的 track），结果从 +15.36% 掉到 **−11.87%**。

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

**所以要控制"哪些物体参与修正"，能改的是 ① 层的规则**（名额、判重排序、按输入侧测量挑选
prompt 集），**不是加一个更好的打分器** —— ① 出局的物体，后面五层从没见过它。

---

## 6. 两处必须分清的机制

### 6.1 环路闭在哪一半

**传播上闭环、估计上开环。** 修正会经累积器带到后面每一帧；但**提案是在任何修正存在之前
一次算完的**，所以第 t 帧的修正改变不了第 t+1 帧怎么估计。

能闭环的只有**参考云**这一半 —— 因为 SAM 只看 RGB、模型 `forward_chunk` **不含位姿**、
depth 是**相机系**的：

| 量 | 修正位姿会改变它吗 |
|---|---|
| SAM 的 mask | **不** |
| depth | **不** |
| **参考云**（靠位姿进世界） | **会** ← 唯一 |

（这条已经实测过：闭环残差降 18% 但轨迹差 3.6%，见实验文档。）

### 6.2 "修点云"有两个

| | (a) 逐实例修点云 | (b) 位姿反馈（**主线**） |
|---|---|---|
| 相机位姿 | **保持 raw** | **被修正** |
| 改的是什么 | 每个实例各修自己的点云 | **放置点的那个位姿** |
| 分支名 | `object_pose_object_only_per_instance` | object-consensus feedback |
| 修正量 | `storage_pose @ inv(raw_pose)`，只作用于该实例的点 | 全部物体投票出的每帧共识 |

**(b) 里点云一个点都没改** —— 同一批相机系点，被不同的轨迹摆到不同的世界位置。
拿 (a) 的逐实例好坏去推断它在 (b) 的共识里话语权多大是错的。

---

## 7. 判据：七条，事前声明

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

**第 6 条（sim3）为什么必要**：direct ATE 单看不够 —— 修正可以纯靠吸收一个全局相似变换
（scale/rigid）把它改善，而相对轨迹没变。100 帧那轮**两种失败模式都出现过**。

**第 7 条（旋转）为什么必要**：旋转估计比它要修的噪声底还差（共识旋转误差 0.611° 对
raw 逐帧 RPE rotation 0.318°）。要求它为正，是为了不让"只帮了平移"的修正蒙混过关。

---

## 8. 两个必须小心的测量前提

### 8.1 噪声底有两种，别用一个数当全部

| 误差来源 | 数字 | 怎么测的 |
|---|---|---|
| 分割 + 分析链（**几何固定**） | **1.8e-05** | 三次重跑，`--reuse-if-valid` 复用同一份 stage-1 缓存 |
| **独立 stage-1 重跑** | **≈ 2.4e-03** | 两次完整重跑（清空 `outputs/` 后重建） |

**第二种才是"别人重跑你的实验"会遇到的差**，比第一种大约 **130 倍**
（HorizonStream 在 GPU 上不是逐位确定的）。

### 8.2 小于 0.01° 的旋转角不可信

那个尺度上测量本身有 ~1e-2 的相对漂移。旋转角用 `atan2(sin, cos)` 而非 `acos(trace)` ——
接近单位阵时迹是平的，`acos` 会把浮点噪声放大成零点几度。

---

## 9. 帧窗、代数与命名

- **帧窗**与 **run 目录名**都在 `object_pose_feedback_env.zsh` 里定义一次，所有命令文件 source 它。
  两个窗口的 run 永不互相污染。
- **代数（generation）**是 prompt 集：`GENERATION` 一行切换，词表从 `GENERATION_PROMPTS` 按标签
  查出（**标签和词表不可能不一致**）。
- **帧选择会静默截断**（`positions[:count]`），所以 `opf_require_frame_window` 在任何东西碰到
  GPU 之前就拒绝"场景不够长"的跑法。
