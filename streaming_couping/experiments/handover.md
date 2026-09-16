# 交接文档：SAM 物体锚点修正 HorizonStream 位姿漂移

2026-09-16 · 实习交接

> **先读这份。** 它告诉你现在有什么、怎么跑、什么已经试过并且失败了、以及哪些坑会浪费时间。
> 想深入某一块时，按 §8 的文档地图去查。

---

## 1. 一句话

在**冻结的 HorizonStream** 上，用 **SAM3.1 的 persistent object tracks** 做跨物体共识、
把修正反馈进因果位姿累积器，**位姿与点云产物同时改善，两个帧窗上都通过全部七条预先声明的判据**。

主结果：direct ATE **+14.25%**（100 帧）、**+15.4 ~ 15.6%**（150 帧），sim3 同为正。
全程不训练，不改 HorizonStream backbone / KV / GLA cache，GT 只在所有决策冻结后加载。

**但它的边界很窄**：单场景、单条开发窗口、阈值就在这条窗口上选的、无 held-out 证据。
详见 §6。

---

## 2. 现在的状态

| 项 | 状态 |
|---|---|
| 主结果 | `robust_semantic`，**v1 prompt 集**，100 帧与 150 帧两轮都 GO |
| 最好的单次配置 | `factorized`，但它**不稳定**，不能当主线（`experiments.md` §6） |
| 代码 | 已精简到这条线；旧线（dinov3 / multiclip / v0 / v11 / v21–23 / sam31-auto / temporal）已全部移除 |
| 规模 | commands 9 · scripts 29 · src 70 · tests 28 · **docs 1 · experiments 2** |
| 测试 | **231 通过**，1 个失败（`test_instance_point_consistency`，**在精简之前就存在**，与本线无关） |

---

## 3. 怎么跑

```bash
# 主入口：一条命令跑完一轮（含 GPU）
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 只重读判定 + 两张图 + 点云评测（纯 CPU，几秒）
zsh streaming_couping/commands_check_object_pose_feedback_decision.txt

# 单元测试 + 逐类别解释 + prompt 账本（纯 CPU，只读）
zsh streaming_couping/commands_verify_object_pose_feedback.txt

# 每个 prompt 返回了几条 track、被谁判重、有没有撞名额（纯 CPU）
zsh streaming_couping/commands_show_sam3_candidate_ledger.txt
```

**默认只跑 baseline、只跑一次**（就是产出主结果的那一轮，约占一次 segmentation pass）。
两个东西是显式打开的，各自换来一张表：

| 打开 | 换来 |
|---|---|
| `OBJECT_POSE_FEEDBACK_BRANCHES=baseline,fresh,factorized,...` | 提案侧消融表 |
| `OBJECT_POSE_FEEDBACK_SAM_REPEATS=3` | 噪声底 |

**换帧窗**：改 `object_pose_feedback_env.zsh` 的 `FRAME_COUNT`；读旧窗口用
`OBJECT_POSE_FEEDBACK_FRAME_COUNT=100 zsh ...`。

**换 prompt 集**：改 sweep 顶部的 `GENERATION` 一行 —— 词表从 `GENERATION_PROMPTS` 按标签
查出，所以标签和词表不可能不一致。

---

## 4. 代码在哪

```
streaming_couping/
  commands_*.txt              9 个入口（3 个在 baseline 链上，6 个是读结果的工具）
  object_pose_feedback_env.zsh 帧窗与 run 目录名的单一来源
  src/semantic_mapping/       这条线的主体：几何适配、SAM 适配、提案、共识、门控、重放
  scripts/                    29 个，其中 18 个在 baseline 链上
  tests/                      28 个
  docs/method.md              方法：pipeline、判据、六层筛选
  experiments/                2 份：experiments.md（实验）+ handover.md（本文件）
```

**baseline 链**：`..._branches.txt` → `..._100f.txt` → `evaluate_..._object_only.txt`。
删这三个里的任何一个，主入口直接跑不起来 —— 精简时确认过。

---

## 5. 已经试过并且失败的方向（**最省时间的部分**）

这些都有实测证据，**别重复做**。每条在 `experiments.md` §6 里都有原始数据。

| 方向 | 结论 |
|---|---|
| **换更好的分割** | `oracle_mask`（**用 GT mask 替换 SAM**）的 `prop_err` **反而更差**（0.1010 vs 0.0892）。瓶颈不在 mask 质量 |
| **参考集太旧** | `fresh` 分支把 anchor 年龄从 37.5 降到 8、相关性从 0.79 降到 0.17，**提案误差纹丝不动**（0.0893→0.0891）。那个相关不是因果 —— anchor 旧 ⟺ 处于序列后半段 |
| **旋转平移解耦** | `factorized` 能全过，但**跨配置从 +3% 摆到 +19%**（100 帧 v1 +2.99% 不过 → 150 帧 v1 +18.64% GO → 100 帧 v3 +19.12% 差一条 → 150 帧 v3 NO_GO）。**不是更好的配置，是不稳定的配置** |
| **只修平移** | `translation_only` **更差**：强制 ω=0 后 sim3 变负。旋转与平移在联合解里耦合 |
| **加 prompt** | `table` 在 SAM 侧**一条 track 都没回**（而 GT oracle 给它出了约 99 条提案）；`cabinet` 回来了 15 条，把全局名额占满，还把 `wardrobe` 判重挤掉，结果 **−11.87%** |
| **闭环 / 流式** | 把修正轨迹喂回去重解提案：**残差降 18.0%**（空对照 +0.1%），但**轨迹差 3.6%**，五变体全 NO_GO。加上原理限制 → **真流式不值得做** |
| **换优化目标** | 目标函数对**共模误差免疫**：`P` 和 `Q` 出自同一条 depth+pose 流水线，共同误差相减时抵消 —— loss 看不见它，而它恰恰贡献绝对位姿误差 |

**`S_geo`（几何置信度）是反预测的**（类内 ρ **+0.342**），`S_sem` 也是（**+0.634**）。
主方法只用 `S_sem` 加权、不用 `S_geo`，**差别全在这里**：多乘 `S_geo` 让增益从 +14.25% 减半到 +7.83%。

---

## 6. 还没回答的问题

| 问题 | 状态 |
|---|---|
| **第二个场景** | **没有**。所有结果来自 `00a231a370` 一条场景。这是最该补的一件事 |
| `anchor_rho` 从 0.79 掉到 0.09 | **未解释**。它在 100 帧上是最强预测因子，窗口变长后理应保持，却消失了 |
| **哪个物体会赢，事前判不出** | 两个为"物体质量"设计的分数都是反预测的，`reliable` 硬拒零区分度。有效的两个（`consensus_inlier`、帧级 gate）都要求**先有一群物体在投票** |
| 16 个名额 + 判重按出生帧 | **已知是机制**（v5 账本实测到了），**没有改**。这是控制"哪些物体参与"唯一能改的地方 |
| v5 的归因 | **分不清**是丢 `wardrobe` 还是加 `cabinet` 造成的 —— 两者同时发生。要分开需让 cabinet 进来而 wardrobe 不丢 |

---

## 7. 交接时必须知道的坑

1. **不要删 `outputs/`。** 删了会失去可比性：**两次独立 stage-1 跑的 d_ATE 差约 0.24 个百分点**
   （清空后重建实测：+15.60% → +15.36%），比很多分支差异还大。而且对照没了就永远分不出来。
   sweep 会把新代的几何缓存**从上一代复制**，所以两代天然共享同一次 stage-1 —— 删了就失去这个。

2. **prompt 列表不是可加的。** 加一个词可能**挤掉**已有的词 —— 两个机制：跨 prompt 判重（按
   **出生帧**排序，重叠就丢后到的）、全局 **16 条**名额（所有词共享）。加词是摊薄名额，不是多几个检测器。

3. **"修点云"有两个，别混。**
   - **(a) 逐实例修点云**：相机位姿保持 raw，每个实例各修自己的点云（`object_pose_object_only_per_instance`）。
   - **(b) 位姿反馈**：修的是**相机轨迹**，点云一个点都没改 —— 改的是**把它放进世界的那个位姿**。
   主线是 (b)。拿 (a) 的逐实例好坏去推断它在 (b) 的共识里话语权多大是错的。

4. **噪声底有两种，别用一个数当全部。** 1.8e-05 是"共享同一次几何缓存"的；跨独立 stage-1 是 **2.4e-03**。

5. **不要用 ICP loss 判断好坏。** 判据只有位姿指标（七条，见 `../docs/method.md` §7）。loss 下降不代表
   位姿变好 —— V3 那次 loss 降 58% 而 GT 指标平坦。

6. **报告里 150 帧的数来自两次不同的 stage-1 跑**（+15.60% 与 +15.36%），章节里都标注了是哪一次。
   引用时注意。

7. **静态导入分析在这条仓库里不可靠。** 精简时它错了三次（漏相对导入、惰性正则抓错标识符、
   把 `recovery`/`semantic_map`/`object_memory` 判成不可达）。判断"旧线还是当前"要看 **docstring**。

---

## 8. 文档地图

**全仓库只有三份文档**（外加 `readme.md`）：

| 想知道什么 | 看哪份 |
|---|---|
| **方法**：pipeline 怎么搭、每步做什么、判据是什么、六层筛选 | [`../docs/method.md`](../docs/method.md) |
| **实验**：做过什么、结果是什么、什么失败了、早期弯路 | [`experiments.md`](experiments.md) |
| **交接**：现在什么状态、怎么跑、坑在哪、下一步 | 本文件 |

`experiments.md` 内部的定位：

| 章节 | 内容 |
|---|---|
| §0–§4 | 结论、位姿主结果、点云传导、环路验证、不依赖单一物体 |
| §5 | 消融：共识侧 5 变体 + 提案侧 5 分支（每个分支检验什么假设） |
| §6 | **已否证的方向**（含闭环、加 prompt 的机制细节） |
| §7 | 适用范围 |
| §8 | 早期实验（这条线的前身，别重复） |
| §9 | 未解释的观察（`anchor_rho`） |
| §10 | 复现命令 |

**这三份文档合并自原来 16 份**，被合并的原文在 git 历史里可查。

---

## 8b. 图

**每轮跑完产出两张图**（CPU，几秒），在 `<run>.baseline/object_pose_feedback/` 下：

| 图 | 画的是 |
|---|---|
| `pose_comparison.png` | 轨迹俯视 + 逐帧平移/旋转误差，GT / raw / 修正后三条线，RMSE 写在标题里 |
| `object_cloud_comparison.png` | 每个物体的点云，**同一批点用三条轨迹摆三次** —— 只有位姿变 |

---

## 9. 如果要继续，我会先做这两件

1. **第二个场景。** 现在 n=1，而且是开发窗口。这是把结论从"这条窗口上成立"变成"这个方法成立"
   的唯一途径，其他都是枝节。
2. **`anchor_rho` 那个消失。** 它在 100 帧上是最强的预测因子（`fresh` 分支就是为它做的），
   到 150 帧却归零。要么是测量问题，要么背后有个没看见的东西 —— 两者都值得知道。
