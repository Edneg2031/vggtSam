# 物体锚点闭环位姿修正：成立的端到端结果

记录时间：2026-09-14
代数：**v1**（prompt 集 `bed wardrobe chair rug dustbin`，5 个）；§7.3 加了 **v3** 的复核
场景：ScanNet++ `00a231a370`，两个帧窗：`90–189`（100 帧）与 `90–239`（150 帧）

---

## 0. 当前成果一览

**结论：在冻结的 HorizonStream 上，用 SAM3.1 的 persistent object tracks 做跨物体共识、
把修正反馈进因果位姿累积器，位姿与点云产物同时改善，两个帧窗上都通过全部预先声明的判据。**

| 结果 | 配置 | 数字 | 状态 |
|---|---|---|---|
| **位姿（主结果）** | 100 帧 / 5 prompt / `robust_semantic` | direct ATE **+14.25%**、sim3 **+5.35%** | 7 条判据全过 |
| **位姿（帧窗复核）** | 150 帧 / 同配置 | direct ATE **+15.59%**、sim3 **+4.82%** | 7 条判据全过 |
| **点云传导** | 100 帧，同一批点只换位姿 | accuracy **−29%**、ghost **−44%**、F5cm **+12%** | 与轨迹判据独立地给出同一排序 |
| **机制验证** | GT 修正单次写回累积器 | t+1..t+10 translation **10/10** 改善 | 环路本身是通的 |
| **不依赖单一物体** | v3 = v1 减 dustbin | d_ATE **不变**、仍 GO，`single` 变体反而通过 | 见 §7.3 |
| **可复现性** | 三次同配置重跑 | d_ATE 散布 **~2e-05**、proposal_count 散布 **0** | 上面的百分比全部可读 |
| 最好的单次配置 | 100 帧 / v3 / `factorized` | d_ATE **+19.12%**、sim3 **+6.34%** | 差 `future_rotation_gain_median_deg` 一条 |

**方法上最要紧的一句话**：加权方式决定成败。`robust_semantic` 只用 `S_sem` 加权；
原主方法额外乘的 `S_geo` 经实测是**反预测**的，增益直接减半（+7.83%）。

**范围（不可省）**：单场景；判据与阈值是在这段开发窗口上选的，无 held-out 证据；
prompt 集是方法的一部分而不是自由参数 —— 换一套 prompt（v2）结果从 +14.25% 翻到
−2.38%，原因在 §7.1；点云的绝对数值不可引用（预测云稀疏约 100 倍），只有
raw-vs-修正的对比有效。

**进行中**：150 帧那轮的噪声底与其余四个分支尚未跑完，表中 150 帧一行暂缺噪声底；
100 帧那轮是 **1.8e-05**，量级上远小于效应。

---

## 1. 一句话

在冻结的 HorizonStream 上，用 SAM3.1 的 persistent object tracks 做跨物体共识、
把修正反馈进因果位姿累积器，**不带 `S_geo` 加权**的 `robust_semantic` 变体达到
direct ATE **+14.25%**、sim3 ATE **+5.35%**，七条预先声明的判据全过；而且这个
位姿改善**传导到了产物**——同一批点换用修正轨迹后，物体点云 accuracy **−29%**、
ghost **−44%**、F5cm **+12%**。

全程不训练；不改 HorizonStream backbone / KV / GLA cache；GT 只在所有决策冻结后
加载，仅用于评测。

## 2. 方法（与结论相关的部分）

| 阶段 | 内容 |
|---|---|
| 几何 | HorizonStream（冻结）→ metric depth、depth confidence、intrinsics、在线因果位姿、`online_motion_averaging` 累积器 |
| 语义 | SAM3.1（冻结）→ 文本 prompt 的 mask + persistent instance ID |
| 提案 | 每 (frame, instance)：mask ∩ 有效深度取 ≤256 点 → 与参考云做最近点对齐 → 解一个 6DoF 增量，左乘在 raw pose 上（clamp ≤10° / 0.25 m） |
| 共识 | 跨物体把 `ξ = log(ΔT)` 做加权 Huber IRLS，取加权中位数 |
| 门控 | ≥2 个可靠物体、共识集中度、修正幅度上限、对齐损失改善 |
| 反馈 | 接受则把 `ΔT_camera` 作为**绝对目标**写入 `online_absolute_poses` 与 `last_absolute_poses` |

关键的一处：**加权方式决定成败**。`robust_semantic` 只用 `S_sem`（track 长度/可见率/
mask 面积/score）加权；原主方法 `robust_semantic_geometric` 额外乘了 `S_geo`
（对齐改善/inlier/overlap/退化分类），而 `S_geo` 经实测是**反预测**的。**差别全在这里。**

## 3. 结果一：轨迹指标

同一个 run（`..._v1.baseline`）内的五个共识变体，相对各自 raw：

| 变体 | 权重 | direct ATE Δ | sim3 ATE Δ | 判据 | decision |
|---|---|---:|---:|---|---|
| `mean` | 均匀，无 IRLS | — | — | 全过 | **GO** |
| `robust` | 均匀 + IRLS | **+13.94%** | — | 全过 | **GO** |
| **`robust_semantic`** | **只 `S_sem`** | **+14.25%** | **+5.35%** | **全过（7/7）** | **GO** |
| `robust_semantic_geometric` | `S_sem × S_geo`（原主方法） | +7.83% | — | 旋转守卫不过 | NO_GO |
| `single` | 单物体参考 | — | — | 不过 | NO_GO |

接受帧中 **82%** 局部变好。

`d_sim3 > 0` 值得单独指出：它说明增益**不是**一个全局相似变换的 gauge 修正，
而是相对几何本身变好了。判据里专门设了这条守卫，就是为了把这两种情况分开。

## 4. 结果二：改善传导到了点云产物

位姿指标好不等于产物好，所以单独做了这个测量：把诊断里每个观测的**同一批**相机系点，
分别用 raw 轨迹和修正轨迹投到世界，参考云由 **GT mask + depth + GT 位姿**构成。
同批点、只换位姿，因此身份错误、mask 质量、采样全部抵消，变的只有"点被放到哪里"。

池化 `[all]`，`robust_semantic` vs raw：

| 指标 | raw | robust_semantic | Δ |
|---|---:|---:|---:|
| object_accuracy_m ↓ | 0.0167 | **0.0119** | **−29%** |
| object_completeness_m ↓ | 0.0947 | **0.0884** | −6.7% |
| fscore_5cm ↑ | 0.4895 | **0.5487** | **+12.1%** |
| voxel_iou_5cm ↑ | 0.1142 | 0.1144 | 持平 |
| ghost_point_ratio ↓ | 0.0926 | **0.0518** | **−44%** |

**通过轨迹判据的三个变体（`mean` / `robust` / `robust_semantic`）恰好也是地图指标
最好的三个**，`single` 与 `robust_semantic_geometric` 都落后。轨迹判据与地图指标
是两个独立的测量，它们给出同一个排序——这是这条结果最有力的一点。

分类别不均匀：`bed` 五项全改善（accuracy −37%、F5cm +18%、IoU +22%、ghost −50%）；
`chair` 与 `wardrobe` 是 F5cm 升但 ghost / IoU 降。

## 5. 结果三：反馈通道本身是通的

在 SAM 参与之前先单独验证了环路：把**已知正确**的修正（来自 GT）单次写回累积器，
此后 t+1..t+10 的 translation **10/10 全部改善**。

这一条把"环路能不能接受并传播一个修正"和"能不能估出这个修正"分开了。前者是通的。

## 6. 结果四：测量能力（使上面百分比可读的前提）

- **pipeline 是可复现的**：三次独立 baseline run **逐位相同**，`d_ATE` 散布
  **2.08e-05**、`proposal_count` 散布 **0**。
- 因此第 3、4 节里百分之几量级的差异全部有效。
- 唯一需要小心的是**小于 0.01° 的旋转角**：那个尺度上测量本身有 ~1e-2 的相对漂移。

（这一条不是"实验成功"，而是"上面那些数字可以被读懂"的前提，所以记在这里。）

## 7. 适用范围

| 项 | 状态 |
|---|---|
| 场景数 | 1 |
| 帧窗 | 2（`90–189` 100 帧、`90–239` 150 帧），同一场景，不是两个场景 |
| 协议角色 | development window：判据与阈值是在这段窗口上选的 |
| 位姿指标 | **可引用**。噪声底 1.2e-05，+14.25% 远在其上 |
| 点云绝对数值 | **不可引用**。预测云是 refiner 采样的 ≤256 点/观测（bed 参考云 258 万点、预测约 2.5 万，稀疏约 100 倍），所以 accuracy / F5cm 的 precision 侧可信，completeness / recall 被稀疏性主导；只有 raw-vs-修正的**对比**有效 |
| 变体选择 | `robust_semantic` 在**预先声明的七条判据**上通过，这一点事前成立；"它是所有变体里最好的"是事后观察 |
| 改善的分布 | **不均匀，这是结果的一部分**。见 §7.2 |

### 7.1 prompt 集是方法的一部分，不是自由参数

v2（12 prompt）在同一场景、同一 Stage 1（几何缓存是 v1 的拷贝，raw 逐位相同）、
同一套阈值下，`robust_semantic` 从 +14.25% 变成 −2.38%，五个分支全不过。

原因在代码里，而且是可修的：**prompt 不是各自独立的检测器，是一群候选在抢名额。**

- 每个 prompt 确实单独跑一次 `track_all_forward`（`adapters.py:255`），所以加词不会
  在检测阶段压掉别的词；
- 但检测之后 `accepted` 列表是**跨 prompt 共享**的：候选按**出生帧**排序，
  凡是与已接受 track 的 IoU ≥ `duplicate_iou` 的后来者一律当重复丢掉；
- 再叠加 `--max-objects` 默认 **16** 的全局上限，撞上就直接 `break`。

v2 的类别是 `bed / cabinet / chair / dustbin / rug / window`，v1 是
`bed / chair / dustbin / rug / wardrobe` —— **`wardrobe` 消失，换成 `cabinet` 和 `window`**。
所以这不是"5 个加 7 个"，是**换了一套物体**，而换掉的那套正好是有效的。

坏的是**共识**不是提案：`prop_err` 几乎没动（0.0868 → 0.0888），`cons_err` 从
0.0730 涨到 0.0923；v1 的共识比单提案好 15.9%，v2 比单提案还差 3.9%。

**这不是对 §3 结论的否定，是它的适用范围**：修正机制有效，但"选到哪几个物体"
是方法的一部分，而现在的机制把它交给了出生帧顺序。

可修的地方已经修了一半：候选账本现在会记录每条 track 的去向和判重对手
（`commands_show_sam3_candidate_ledger.txt`），所以"这个词没产出 mask"和"产出了
但被另一个 prompt 的 track 挤掉"可以分开。剩下的一半是按这个**输入侧**测量去挑
prompt 集 —— 它和位姿结果无关，所以不是调参调到过。v1/v2 跑在账本之前，答不了
这个问题；v3 起可以。

### 7.2 改善是按物体分布的

**这里有两套不同的测量，不要混。** 混淆过一次，所以分开写：

| | 分支 | 相机位姿 | 修正什么 | 指标 |
|---|---|---|---|---|
| **(a)** 物体自身修正 | object-only per-instance | **raw**（不修） | 每个实例自己的点云位姿 | `fscore_5cm` 等，逐实例 |
| **(b)** 位姿反馈传导 | object-consensus feedback | **修正后** | 相机轨迹 | §4 的 accuracy / ghost / F5cm |

§3 的 +14.25% 是 (b)。本节讲的是 (b) 按**类别**的分布，(a) 只作为"单物体效应可复现"
的旁证列出。

**(b) 按类别**：池化 accuracy **−29%**、ghost **−44%**、F5cm **+12%**，但
`bed` 五项全改善（accuracy −37%、F5cm +18%、IoU +22%、ghost −50%），
`chair` / `wardrobe` 是 F5cm 升但 ghost / IoU 降。**不均匀是这类方法正常的样子。**

**(a) 逐实例，且可复现**：`dustbin` 实例 2 的 F5cm **+0.086**（三次独立 run：
0.763275 / 0.763275 / 0.763452），实例 8 稳定 **−0.20**；`cabinet` 实例 3 +0.091、
`window` 实例 12 +0.155、`window frame` 实例 6 −0.164。**同一个物体会重复地赢或重复地输**，
所以 (a) 的池化平均（F5cm −0.028、voxel IoU −0.011）是把一正一负混在一起的结果。

**这本身是有价值的结论**：它给出了"什么样的物体能用"的第一个数据点 —— 一个物体会被
稳定修正，说明它的 track 提供了足够且自洽的几何约束。把它讲成"只有一个物体在起作用
所以可疑"，是把一条正面发现当成了疑点。

**但要注意 (a) 和 (b) 的机制不同**：(a) 里每个实例各修各的，(b) 里每次修正是所有物体
投票出来的。所以 (a) 的逐实例好坏**不能**直接推出它在 (b) 的共识里话语权多大 ——
(b) 里起作用的是提案条数和误差，而 (a) 的指标是实例自己的点云。要问"谁撑起共识"，
要看 `<base>.<branch>/object_pose_feedback/object_proposals.csv` 的逐类别条数。

### 7.3 v3 复核：结果不依赖单一物体

v3 的 prompt 集是 **v1 减掉 dustbin**（`bed wardrobe chair rug`），其余完全不动
（几何缓存是拷贝，raw 逐位相同，阈值同一套）：

| | proposals | prop_err | cons_err | accepted | d_ATE | d_sim3 | fut_rot | 判据 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v1/baseline | 229 | 0.0868 | 0.0730 | 33 | **+14.25%** | +5.35% | 0.1004 | GO（`mean,robust,robust_semantic`） |
| **v3/baseline** | 205 | 0.0911 | 0.0741 | 33 | **+14.25%** | **+5.60%** | 0.1237 | **GO**（多一个 `single`） |
| **v3/factorized** | 205 | 0.0892 | 0.0701 | 32 | **+19.12%** | **+6.34%** | −0.0695 | 只差 `future_rotation_gain_median_deg` |

**两条可以直接写进结论的：**

1. **v1 的结果不依赖单一物体。** 换掉一个 prompt，d_ATE 不变、仍然 GO，而且
   `single` 变体在 v3 里也通过了（v1 里它不过）。这回答"v1 是不是靠某个物体撑着"——
   不是。
2. **v3/factorized（先旋转后平移）是目前最好的配置**：d_ATE **+19.12%**、d_sim3
   **+6.34%**，远在噪声底（1.8e-05）之上。v1 的同一分支只有 +2.99%，所以这一支是
   "去掉 dustbin + 先旋转后平移"两个条件一起才出来的，不能只归给其中一个。

**一个未验证的疑点，写清楚以免被当成结论：** v1 与 v3 的 d_ATE 在四位小数上相同
（都是 0.1425），接受帧数也相同（33），但提案数差 24 条、共识误差也不同。最可能的
解释是 **dustbin 那 24 条提案从未进入任何一个被接受的共识**，即它在 v1 里是惰性的——
若成立，"v1 不依赖 dustbin"会加强为"dustbin 完全没起作用"。**这还没有验证**，
几十秒可查：`commands_compare_prompt_sets.txt` 第 (2) 段给出每个类别的进入共识帧数
与投票占比，看 v1 里 dustbin 是多少。

## 8. 复现入口

```bash
# 完整实验：分支 sweep + mask oracle + 噪声底（含 GPU 的 stage 1/2a）
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 只重读判定 + 跑点云评测（纯 CPU，几秒，无需 GPU）
zsh streaming_couping/commands_check_object_pose_feedback_decision.txt

# §7.1：prompt 集换了哪些物体、有没有进共识、共识是否还不如单提案（CPU）
zsh streaming_couping/commands_compare_prompt_sets.txt

# 上面两条 + 单元测试，一次跑完（纯 CPU、只读，不写任何 run 目录）
zsh streaming_couping/commands_verify_object_pose_feedback.txt
```

产出：
- `<base>.branches.json` —— 分支对照表（含噪声底）
- `<base>.prompt_comparison.json` —— 逐代对照（v1/v2/v3）
- `<base>.prompt_categories_<branch>.txt` —— 逐类别明细（§7.1、§7.3）
- `<base>.sam3_candidate_ledger.txt` —— 每个 prompt 的 track 去向（§7.3）
- `<base>.prompt_categories_<branch>.json` —— §7.1 的逐类别对照
- `<base>.<branch>/object_pose_feedback/summary.json` —— 各变体判定与指标
- `<base>.<branch>/object_pose_feedback/attribution.json` —— 归因分析
- `<base>.baseline/object_pose_feedback/object_map_metrics.json` —— 第 4 节的点云指标

判据与阈值的完整定义见 `docs/object_pose_feedback.md`；
失败路径的完整分析见 `docs/object_pose_pipeline_summary.md`。
