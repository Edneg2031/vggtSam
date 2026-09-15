# 物体锚点闭环位姿修正：成立的端到端结果

记录时间：2026-09-14
代数：**v1**（prompt 集 `bed wardrobe chair rug dustbin`，5 个）；§7.3 加了 **v3** 的复核
场景：ScanNet++ `00a231a370`，两个帧窗：`90–189`（100 帧）与 `90–239`（150 帧）

---

## 0. 当前成果

汇报用的报告在 [`report.md`](report.md)：问题、方法、实验过程、结果、结论。
数字只在报告里写一次，本节不再重复 —— 两份副本迟早会各自漂移。

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

- **同一个 stage-1 缓存下是可复现的**：三次 baseline 重跑 **逐位相同**，
  `d_ATE` 散布 **2.08e-05**、`proposal_count` 散布 **0**。
  **但这三次共用同一份几何缓存** —— 见下面的更正。
- 因此第 3、4 节里百分之几量级的差异全部有效。
- 唯一需要小心的是**小于 0.01° 的旋转角**：那个尺度上测量本身有 ~1e-2 的相对漂移。

**更正（2026-09-15，实测）**：清空 `outputs/` 后完整重跑，`d_ATE` 从 +15.60% 变成
**+15.36%**，accepted 33 → 34。差 2.4e-03，是上面那个 2e-05 的 **约 130 倍**。

原因：三次重跑是 `--reuse-if-valid` **复用同一份 stage-1 几何缓存**，只测了分割与分析链
的方差。**HorizonStream 在 GPU 上不是逐位确定的** —— 两次独立 stage-1 的 raw ATE 差在
第六位小数（0.138966710 vs 0.138965235）。

所以：**小于约 0.25 个百分点的分支差异，只有共享同一次 stage 1 时才可读。** 效应本身
（15 个百分点）远在这个差之上，结论不变。

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

1. **v1 的结果不依赖单一物体 —— 两个窗口都验证过。** 换掉一个 prompt，d_ATE 不变、
   仍然 GO：100 帧 +14.25% 对 +14.25%（`single` 变体在 v3 里还额外通过了），
   150 帧 +15.36% 对 +15.35%（sim3 0.0429 对 0.0474，v3 略好）。这回答
   "v1 是不是靠某个物体撑着"—— 不是。
2. **v3/factorized（先旋转后平移）是目前最好的配置**：d_ATE **+19.12%**、d_sim3
   **+6.34%**，远在噪声底（1.8e-05）之上。v1 的同一分支只有 +2.99%，所以这一支是
   "去掉 dustbin + 先旋转后平移"两个条件一起才出来的，不能只归给其中一个。

**一个未验证的疑点，写清楚以免被当成结论：** v1 与 v3 的 d_ATE 在四位小数上相同
（都是 0.1425），接受帧数也相同（33），但提案数差 24 条、共识误差也不同。最可能的
解释是 **dustbin 那 24 条提案从未进入任何一个被接受的共识**，即它在 v1 里是惰性的——
若成立，"v1 不依赖 dustbin"会加强为"dustbin 完全没起作用"。**这还没有验证**，
几十秒可查：`commands_compare_prompt_sets.txt` 第 (2) 段给出每个类别的进入共识帧数
与投票占比，看 v1 里 dustbin 是多少。

### 7.4 闭环一步:提案层的受控结果

这条 pipeline 是两遍式的 —— 提案全部从 raw 几何一次算完,修正只在之后重放。所以
修正到不了"下一帧的提案怎么解"。**可闭环的只有"参考云"这一半**(SAM 看 RGB、
模型 forward 不含位姿、depth 是相机系的,都不受位姿影响),而参考云依赖位姿,
所以用修正后的轨迹重摆参考云、重解提案,就是闭环的一步。**纯 CPU。**

150 帧 v1 baseline,`robust_semantic` 的修正轨迹当作新基座:

| | 平移残差中位 | 旋转残差中位 |
|---|---:|---:|
| **空对照**(基座不变,base_shift ≈ 2e-6 m) | 0.089297 → **0.089388**(**+0.1%**) | 1.759863 → 1.725483(−2.0%) |
| **闭环**(喂回修正,base_shift 中位 0.025 m) | 0.089297 → **0.073212**(**−18.0%**) | 1.759863 → **1.772997**(+0.7%) |

**空对照是这条结果成立的关键**:重解路径不逐位复现 stage 2a(它从诊断重建观测和
配置),所以 18% 必须先扣掉重建自己的贡献 —— 实测只有 **+0.1%**。所以那 18%
可以归给"把修正喂回去"这个动作。顺带,空对照的 base_shift 是 2e-6 m,说明两种
c2w 表示的 gauge 映射是对的。

**但改善是平移单侧的**:旋转在闭环里 **+0.7%**,而它自己的漂移是 −2.0% ——
比漂移还差约 2.7 个百分点。和"旋转是最差的一半"(共识旋转误差 0.61° 对 raw
RPE rotation 0.318°)一致。

**端到端结果:轨迹变差了。** 让 Stage 2b 接受外部基座(`--base-trajectory`),用**同一条
代码路径**跑第二轮 —— 只差基座和提案:

| | 起点 | 修正后 d_ATE | accepted | 判据 |
|---|---:|---:|---:|---|
| 第一轮 | raw 0.10946 m | **0.09265 m(−15.4%)** | 34 | GO |
| **第二轮** | 基座 0.092648 m | **0.096010 m(+3.6% 变差)** | 23 | **NO_GO**(五变体全不过) |

**残差降 18%,轨迹反而差 3.6%。** 也就是说闭环**确实**携带了第一轮没用到的信息,
而那些提案照着新基座解出来之后把轨迹推坏了 —— 和整份分析反复出现的那个模式一致:
**优化目标下降不等于你关心的指标变好**。

（接受帧 34 → 23 是这个结果的一部分:基座已被修过,"对齐还能否改善"的空间变小,
gate 拒得更多;但被接受的那 23 帧仍然把结果改坏了。）

**所以:闭环在提案层成立、在轨迹层不成立。** 真流式不值得做。

### 7.5 加一个词的两个机制:账本第一次抓到现场

v5 = v1 加 `cabinet`(v2 加过的 7 个词里只有 2 个真产出过 track,`cabinet` 是其一,
而且是那轮提案最多的类别)。同一份 stage-1 缓存,150 帧。

**账本(第一次直接看到机制开火):**

| prompt | accepted | duplicate | over_object_cap |
|---|---:|---:|---:|
| bed | 1 | 0 | 0 |
| **wardrobe** | **0** | **1** | 0 |
| chair | 3 | 0 | 0 |
| rug | 1 | 0 | 1 |
| dustbin | 5 | 0 | 0 |
| **cabinet** | **6** | 0 | **9** |

- **`wardrobe` 被判重丢掉**(0 accepted / 1 duplicate) —— 输给一条出生更早的 track。
  这是当初对 v2 那个推断(「cabinet 认领了衣柜的像素」)的**直接证据**。
- **`cabinet` 一个词回了 15 条 track**(6 accepted + 9 撞在上限外)。
- **accepted 合计 1+0+3+1+5+6 = 16 —— 正好撞满上限**,`rug` 也被砍掉 1 条。

**结果:d_ATE −11.87%,五变体全 NO_GO**,点云图里 `wardrobe` 消失
(`objects=bed, cabinet, dustbin, chair`)。与 v2 的失败一致。

**归因的限制(必须写明):** 丢掉 `wardrobe` 和加进 `cabinet` **同时发生**,所以这一轮
**分不清**是哪一个造成的。要分开,需要让 cabinet 进来而 wardrobe 不丢 —— 那要抬
`--max-objects` 或改判重的排序,是另一个实验。

**一个附带的观察**:v5 是**平移 −11.87%、旋转 +10.08%**(v1 是平移 +15.4%、旋转 +3.9%),
像是拿平移换了旋转,净结果 NO_GO。

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
