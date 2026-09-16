# 物体筛选判据：六层，以及每一层的实测效果

2026-09-16

## 0. 为什么需要这份文档

修正的质量取决于**哪些物体参与了投票**。而物体集合会在**六个不同的层**被改变，
每层各有各的判据 —— 其中**前几层根本不做质量判断**。

这件事在 v5 上第一次被测出来：加一个词 `cabinet`，`wardrobe` 就消失了，结果从
**+15.36% 掉到 −11.87%**。而 `wardrobe` 出局的原因是**出生帧顺序**，与它好不好无关。
（详见 `experiments/object_pose_feedback_go.md` §7.5。）

所以本文把六层逐层列清，并说明每一层的判据**实测有没有用**。

---

## 1. 六层结构

```
① 跟踪层   哪些物体**存在**？        prompt 词表 · 全局名额 · 跨 prompt 判重
              ↓
② 观测层   哪些能**形成提案**？      track score · static · mask 大小 · 几何置信度 · 点数
              ↓
③ 可靠层   哪些**算可信**？          S_sem（语义）· S_geo（几何）
              ↓
④ 共识层   哪些**进中位数**？        ξ=log(ΔT) 的 Huber IRLS → consensus_inlier
              ↓
⑤ 门控层   这一帧的共识**要不要用**？ 物体数 · 集中度 · 幅度上限 · 对齐损失必须下降
              ↓
⑥ 加权层   用了之后**各占多少权重**？  robust_semantic = 只按 S_sem
```

**关键：① 在 ②–⑥ 之前。** 一个物体如果在 ① 就没进来，后面五层**从来没见过它**，
任何质量判据都不会有机会对它说话。

---

## 2. 逐层的具体判据

### ① 跟踪层 —— 决定"存在"

| 判据 | 取值 | 判断的是 |
|---|---|---|
| prompt 词表 | v1 = `bed wardrobe chair rug dustbin` | 文本能否命中 |
| `max_objects_per_prompt` | 16 | 单个词最多回几条 track |
| **`--max-objects`（全局）** | **16** | **所有词加起来最多收几条** |
| 判重 `duplicate_iou` | 0.80 | 两条重叠 track 谁活 —— **按出生帧，先到先得** |
| `min_birth_pixels` | 128 | 出生那一帧 mask 的最小像素 |

**这一层没有任何质量判据。** 排序键是 `(出生帧, prompt 序号, obj_id)`；名额满了就
`break`，后面的候选**连判重都不会被检查**。

### ② 观测层 —— 决定"能否形成提案"

`min_track_score 0.50`、`static_score_threshold 0.20`、`min_mask_pixels 32`、
mask 面积 ≤ 0.85、`min_geometry_confidence 0.30`、`min_points_per_observation 24`、
`min_matches_per_pair 8`、`min_total_matches 16`。

### ③ 可靠层 —— 决定"算不算可信"

两个**规则化**分数（无 learned score）：

**`S_sem`**（`semantic_reliability`）—— 四个分量的加权和：

```
track_length_score   归一化到 [min_track_length, target_track_length]
visibility_score     可见率 / min_visibility_ratio
mask_score           mask 像素 / target_mask_pixels
score_component      (SAM score − 0.5) / 0.5
```

硬拒（→ `low_track_confidence`）：track 太短 / 可见率太低 / SAM score 太低。

**`S_geo`**（`geometric_reliability`）—— 三个分量 + 一个退化阻尼：

```
improvement_score    对齐损失改善 / target_relative_improvement
inlier_score         inlier ratio / target_inlier_ratio
overlap_score        重叠点数 / target_overlap_points
degeneracy_damp      协方差特征值比（volumetric ←→ degenerate），只降权不直接拒
```

硬拒：`degenerate_geometry` / `correction_too_large`（幅度超 10° / 0.25 m）/
`low_geometry_confidence`。

### ④ 共识层 —— 决定"进不进中位数"

把每个提案的 `ξ = log(ΔT)` 做**加权 Huber IRLS**，取**加权中位数**。每个提案被
标上 `consensus_inlier`：离共识太远的被判为离群，降权或剔除。

**这一层是唯一一个"物体互相投票"的判据。**

### ⑤ 门控层 —— 决定"这一帧用不用"

按优先级输出拒绝原因：

```
insufficient_objects（<2 个可靠物体）
  → low_track_confidence → degenerate_geometry / correction_too_large /
    low_geometry_confidence
  → no_consensus → correction_too_large（共识幅度）
  → no_alignment_improvement
```

最后一道用**真实参考云**复算加权 NN 残差：共识修正统一作用到所有贡献物体后，
**残差必须严格下降**。

### ⑥ 加权层 —— 决定"各占多少权重"

主方法 `robust_semantic` **只按 `S_sem`** 加权。原主方法额外乘 `S_geo`，实测**使增益
减半**（+14.25% → +7.83%）。

---

## 3. 实测：哪些判据真的有效

来自 100 帧 baseline 的归因（类内 ρ = 固定类别后的 Spearman；保留 vs 丢弃 = 该筛选
留下的提案与丢掉的中位 GT 误差）：

| 判据 | 数字 | 结论 |
|---|---:|---|
| `S_sem`（`semantic_confidence` 代理） | 类内 ρ = **+0.634** | **反预测** —— 分数越高，提案越差 |
| `S_geo`（`geometry_confidence` 代理） | 类内 ρ = **+0.342** | **反预测** |
| `reliable` 硬拒筛选 | 保留 0.0881 vs 丢弃 0.0810 | **零区分度** |
| **`consensus_inlier`** | 保留 **0.0753** vs 丢弃 **0.1573** | **有效（差 2 倍）** |
| **帧级 gate** | 接受 **0.0405** vs 拒绝 **0.0914** | **有效（差 2.3 倍）** |
| `track_length` | 类内 ρ = **+0.78** | 最强预测因子，但方向是**负的**（track 越老越差），且**事后才有** |

**两个设计来做"物体质量"判断的分数是反的；两个真正有效的都是"一致性"判据。**

---

## 4. 结构性结论：判据的生效顺序

把上面两条合起来看，会得到一个比"再调调阈值"更硬的结果：

**判据分两类，而且它们在时间上不在同一个位置。**

| 类别 | 例子 | 什么时候能算 | 效果 |
|---|---|---|---|
| **物体自身质量** | `S_sem`、`S_geo`、`reliable` | 拿到这个物体就能算 | **反预测 / 零区分度** |
| **跨物体一致性** | `consensus_inlier`、帧级 gate | **必须已经有一群物体在投票** | 有效 |
| **出生顺序** | 名额、判重 | 跟踪时 | 与质量无关，但**决定谁在场** |

于是：

1. **单独给一个物体打分，现有证据说明做不到** —— 两个试过的都是反的。
2. **能用的信号要求先有共识**，所以它救不了"某个物体在共识形成之前就出局"这种情况。
3. **而 v5 的 `wardrobe` 恰恰是第 3 类** —— 它在 ① 层被出生帧顺序淘汰。**即使造出一个
   完美的物体质量判据，也照不到它**，因为它根本没进到判据能生效的地方。

**所以要控制"哪些物体参与修正"，能改的是 ① 层的规则**（名额、判重排序、或者按
输入侧测量挑选 prompt 集），**不是加一个更好的打分器**。

---

## 5. 附：一个可用的输入侧信号

上面说的都是**输出侧**（看提案好不好）。有一个**输入侧**的信号是可测的、而且与位姿
结果无关 —— **每个 prompt 到底返回了几条 track、有没有被丢掉**：

```bash
zsh streaming_couping/commands_show_sam3_candidate_ledger.txt
```

它把每个词的 `accepted` / `duplicate` / `over_object_cap` 分开记，以及"**什么都没返回**"
和"**返回了但全在出生门被丢**"这两种不同的失败（修法相反：前者换词，后者查阈值）。

实测过的一次：`table` 在 SAM 侧**一条 track 都没有**，而同一轮的 GT-mask oracle 给它
出了约 **99 条提案** —— 同一个词、同一个场景、同一份几何。**天花板在分割器的文本
响应上，不在物体可见性上。**

---

## 6. 相关文档

- `experiments/report.md` —— 汇报用：§4.3 提案侧消融、§5.2 prompt 集的范围
- `experiments/object_pose_feedback_go.md` §7.1 / §7.5 —— 机制的第一手记录与账本
- `docs/object_pose_feedback.md` —— 实现与判据的完整定义
- `docs/object_pose_pipeline_summary.md` —— 失败路径与归因分析（§3.5 首次给出反预测）
