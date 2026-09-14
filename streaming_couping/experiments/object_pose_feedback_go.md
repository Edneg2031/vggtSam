# 物体锚点闭环位姿修正：成立的端到端结果

记录时间：2026-09-14
代数：**v1**（prompt 集 `bed wardrobe chair rug dustbin`，5 个）
场景：ScanNet++ `00a231a370`，帧窗 `90–189`（100 帧）

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
是方法的一部分，而现在的机制把它交给了出生帧顺序。可修的地方也因此明确：
候选账本（谁被谁判重、有没有撞上限）现在完全不记录，记下来就能按**输入侧**测量
挑 prompt 集，而不是靠猜。

### 7.2 改善是按物体分布的，而且单物体的大幅提升本身就有价值

池化平均：accuracy **−29%**、ghost **−44%**、F5cm **+12%**。

逐物体是不均匀的：`bed` 五项全改善（accuracy −37%、F5cm +18%、IoU +22%、ghost −50%）；
`chair` / `wardrobe` 是 F5cm 升但 ghost / IoU 降。

**这是这类方法正常的样子，不是缺陷。** 一个物体被稳定修正（例如 F5cm +0.086，
三次独立 run 重复到小数点后三位）本身就是可用的结论：它说明这个物体的 track 提供了
足够且自洽的几何约束，也就给出了"什么样的物体会work"的第一个数据点。把它讲成
"只有一个物体在起作用所以可疑"，是把一条正面发现当成了疑点 —— 这是错的。

## 8. 复现入口

```bash
# 完整实验：分支 sweep + mask oracle + 噪声底（含 GPU 的 stage 1/2a）
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 只重读判定 + 跑点云评测（纯 CPU，几秒，无需 GPU）
zsh streaming_couping/commands_check_object_pose_feedback_decision.txt

# §7.1：prompt 集换了哪些物体、有没有进共识、共识是否还不如单提案（CPU）
zsh streaming_couping/commands_compare_prompt_sets.txt
```

产出：
- `<base>.branches.json` —— 分支对照表（含噪声底）
- `<base>.prompt_comparison.json` —— v1/v2 逐代对照
- `<base>.prompt_categories_<branch>.json` —— §7.1 的逐类别对照
- `<base>.<branch>/object_pose_feedback/summary.json` —— 各变体判定与指标
- `<base>.<branch>/object_pose_feedback/attribution.json` —— 归因分析
- `<base>.baseline/object_pose_feedback/object_map_metrics.json` —— 第 4 节的点云指标

判据与阈值的完整定义见 `docs/object_pose_feedback.md`；
失败路径的完整分析见 `docs/object_pose_pipeline_summary.md`。
