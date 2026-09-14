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

## 7. 边界的边界：这个结论不能外推到哪里

写清楚，是因为正面结果最容易被当成承诺：

| 项 | 状态 |
|---|---|
| 场景数 | **1**。没有第二个场景的证据 |
| 协议角色 | **development window**。判据与阈值是在这段窗口上选的 |
| held-out 证据 | **无** |
| 点云绝对数值 | **不可引用**。预测云是 refiner 采样的 ≤256 点/观测（bed 参考云 258 万点、预测约 2.5 万，稀疏约 100 倍），所以 accuracy / F5cm 的 precision 侧可信，**completeness / recall 被稀疏性主导**，只有 raw-vs-修正的**对比**有效 |
| 类别一致性 | **不均匀**。`chair` / `wardrobe` 的 F5cm 升但 ghost / IoU 降 |
| 变体选择 | `robust` / `robust_semantic` 是在看过结果后确定为主变体的。它在**预先声明的七条判据**上通过，这一点是事前成立的；但"它是所有变体里最好的"是事后观察 |
| 代数 | 数字来自 **v1（5 prompt）**。v2（12 prompt）尚未运行，本记录不包含任何 v2 数字 |

## 8. 复现入口

```bash
# 完整实验：分支 sweep + mask oracle + 噪声底（含 GPU 的 stage 1/2a）
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 只重读判定 + 跑点云评测（纯 CPU，几秒，无需 GPU）
zsh streaming_couping/commands_check_object_pose_feedback_decision.txt
```

产出：
- `<base>.branches.json` —— 分支对照表（含噪声底）
- `<base>.prompt_comparison.json` —— v1/v2 逐代对照
- `<base>.<branch>/object_pose_feedback/summary.json` —— 各变体判定与指标
- `<base>.<branch>/object_pose_feedback/attribution.json` —— 归因分析
- `<base>.baseline/object_pose_feedback/object_map_metrics.json` —— 第 4 节的点云指标

判据与阈值的完整定义见 `docs/object_pose_feedback.md`；
失败路径的完整分析见 `docs/object_pose_pipeline_summary.md`。
