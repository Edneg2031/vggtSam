# 物体锚点闭环位姿修正 —— 成果总结

2026-09-14 · 场景 ScanNet++ `00a231a370` · 帧窗 `90–189`（100 帧）与 `90–239`（150 帧）

> 这是**汇报用**的总结。完整的实验记录、失败路径与归因见
> [`object_pose_feedback_go.md`](object_pose_feedback_go.md)；
> 失败的假设与为什么失败见 [`../docs/object_pose_pipeline_summary.md`](../docs/object_pose_pipeline_summary.md)。

## 一句话

在**冻结的 HorizonStream** 上，用 **SAM3.1 的 persistent object tracks** 做跨物体共识、
把修正反馈进因果位姿累积器，**位姿与点云产物同时改善，两个帧窗上都通过全部预先声明的判据。**

全程不训练；不改 HorizonStream backbone / KV / GLA cache；GT 只在所有决策冻结后加载，
仅用于评测。

## 结果

| 结果 | 配置 | 数字 | 状态 |
|---|---|---|---|
| **位姿（主结果，可重复）** | 100 帧 → 150 帧，同配置 `robust_semantic` | direct ATE **+14.25% → +15.60%**、sim3 **+5.35% → +4.82%** | **两次都 7 条判据全过** |
| **点云传导** | 100 帧，同一批点只换位姿 | accuracy **−29%**、ghost **−44%**、F5cm **+12%** | 与轨迹判据独立地给出同一排序 |
| **机制验证** | GT 修正单次写回累积器 | t+1..t+10 translation **10/10** 改善 | 环路本身是通的 |
| **不依赖单一物体** | v3 = v1 减掉 `dustbin` | d_ATE **不变**、仍 GO，`single` 变体反而通过 | 见记录 §7.3 |
| **可复现性** | 三次同配置重跑 | d_ATE 散布 **~2e-05**、proposal_count 散布 **0** | 上面的百分比全部可读 |
| **判据稳定性** | 三次同配置重跑（150 帧） | d_ATE 散布 **4.5e-05**、proposal_count 散布 **0** | 上表所有分支差异都远超它 |
| ⚠️ `factorized`（先旋转后平移） | 同分支跨窗口/跨 prompt 集 | 100 帧 v1 **+2.99%**（不过）→ 150 帧 v1 **+16.66%**（全过）→ 100 帧 v3 **+19.12%**（差一条） | **不稳定，不能当主线结果** |

`d_sim3 > 0` 单独值得说：它说明增益**不是**全局相似变换的 gauge 修正，而是相对几何本身变好了。
判据里专门设了这条守卫，就是为了把这两种情况分开。

## 方法（与结论相关的部分）

| 阶段 | 内容 |
|---|---|
| 几何 | HorizonStream（冻结）→ metric depth、depth confidence、intrinsics、在线因果位姿、`online_motion_averaging` 累积器 |
| 语义 | SAM3.1（冻结）→ 文本 prompt 的 mask + persistent instance ID |
| 提案 | 每 (frame, instance)：mask ∩ 有效深度取 ≤256 点 → 与参考云做最近点对齐 → 解一个 6DoF 增量，左乘在 raw pose 上（clamp ≤10° / 0.25 m） |
| 共识 | 跨物体把 `ξ = log(ΔT)` 做加权 Huber IRLS，取加权中位数 |
| 门控 | ≥2 个可靠物体、共识集中度、修正幅度上限、对齐损失改善 |
| 反馈 | 接受则把 `ΔT_camera` 作为**绝对目标**写入 `online_absolute_poses` 与 `last_absolute_poses` |

**最要紧的一句话：加权方式决定成败。** `robust_semantic` 只用 `S_sem`（track 长度 /
可见率 / mask 面积 / score）加权；原主方法 `robust_semantic_geometric` 额外乘了
`S_geo`（对齐改善 / inlier / overlap / 退化分类），而 `S_geo` 经实测是**反预测**的 ——
增益从 +14.25% 直接减半到 +7.83%。

## 范围（不可省）

| 项 | 状态 |
|---|---|
| 场景数 | **1** |
| 帧窗 | 2 个（100 帧、150 帧），**同一场景的复核**，不是第二个场景 |
| 协议角色 | development window：判据与阈值是在这段窗口上选的，**无 held-out 证据** |
| prompt 集 | **方法的一部分，不是自由参数**。换一套 prompt 结果从 +14.25% 翻到 −2.38% —— 因为分割器对 prompt 的响应不可加：加词会改变哪些物体被跟踪，包括**丢掉本来在跟的**。见记录 §7.1 |
| 点云绝对数值 | **不可引用**。预测云是 refiner 采样的 ≤256 点 / 观测（bed 参考云 258 万点、预测约 2.5 万，稀疏约 100 倍），accuracy / F5cm 的 precision 侧可信，completeness / recall 被稀疏性主导；**只有 raw-vs-修正的对比有效** |
| 改善的分布 | **不均匀**。`bed` 五项全改善；`chair` / `wardrobe` 是 F5cm 升但 ghost / IoU 降。不均匀是这类方法正常的样子，见记录 §7.2 |
| 变体选择 | `robust_semantic` 在**预先声明的七条判据**上通过，这一点事前成立；"它是所有变体里最好的"是事后观察 |

**两个窗口都已跑完**，噪声底 100 帧 **1.8e-05**、150 帧 **4.5e-05**，量级上远小于效应。

**一个未解释的观察，不写进结论**：`anchor_rho`（anchor 年龄与 GT 修正误差的类内相关）
在 100 帧是 **0.79**，到 150 帧掉到 **0.09**。它在 100 帧上是最强的预测因子（`fresh`
分支就是为它做的），窗口变长、anchor 更老（37.5 → 52.5 帧）之后理应保持，不该消失。

**判据之外**：`oracle_mask`（用 GT mask 替换 SAM）在两个窗口上 `prop_err` 都**比 SAM 差**
（100 帧 0.1023 vs 0.0845；150 帧 0.1010 vs 0.0892）—— 换 GT mask 不帮忙，两个窗口一致。

## 复现

```bash
# 完整实验：分支 sweep + mask oracle + 噪声底（含 GPU）
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 只重读判定 + 跑点云评测（纯 CPU，几秒）
zsh streaming_couping/commands_check_object_pose_feedback_decision.txt

# 单元测试 + 逐类别解释 + prompt 账本（纯 CPU，只读）
zsh streaming_couping/commands_verify_object_pose_feedback.txt
```
