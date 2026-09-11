# 物体锚点位姿修正：方法、实验与失败分析

更新时间：2026-09-11

## 1. 方法（当前 pipeline）

输入一段 RGB 与文本 prompt，目标是用 SAM3.1 的 persistent object tracks 修正
HorizonStream 的累计位姿漂移，并把修正反馈到后续帧。

| 阶段 | 内容 |
|---|---|
| 几何 | HorizonStream（冻结）→ metric depth、depth confidence、intrinsics、在线因果位姿（w2c）、`online_motion_averaging` 因果累积器 |
| 语义 | SAM3.1（冻结）→ 文本 prompt 的 mask + persistent instance ID |
| 提案 | 每 (frame, instance)：mask ∩ 有效深度取 ≤256 点 → 与参考云（帧 0–4 的固定 anchor，权重 1.0；近期 history，权重 0.5）做最近点对齐 → 解一个 6DoF 增量，左乘在 raw pose 上（clamp ≤10° / 0.25 m） |
| 验证 | `S_sem`（track 长度/可见率/mask 面积/score）× `S_geo`（对齐改善/inlier/overlap/退化分类）→ 跨物体把 `ξ=log(ΔT)` 做加权 Huber IRLS 共识 |
| 门控 | ≥2 个可靠物体、共识足够集中、修正幅度在限内、aggregate 对齐损失改善 → ACCEPT |
| 反馈 | 接受则把 `ΔT_camera` 作为**绝对目标**写入 `online_absolute_poses` 与 `last_absolute_poses` |

不训练；不改 HorizonStream backbone / KV / GLA cache；GT 只在所有决策冻结后加载，
仅用于评测。

## 2. 做过的实验与结果

### 2.1 只修物体点云（相机位姿保持 raw）

| 实验 | 结果 |
|---|---|
| V1 object point alignment | **实际没应用修正**（旋转 0.0，平移 5.8e-17），raw 与 aligned 指标完全相同 |
| V3 instance point alignment | 内部 RMSE 0.0369 → 0.0108（**−58%**），但 GT 指标几乎不动（F5cm +0.0004、voxel IoU −0.0013、ghost +0.0014） |
| shared object-only 6DoF（100 帧） | **bed 四项全改善**：accuracy 0.0709→0.0587、completeness 0.0908→0.0843、F5cm 0.3975→**0.5244**、ghost 0.2346→**0.1763**；但聚合 accuracy +0.0030、ghost +0.0149 变差 |
| per-instance object-only（100 帧，start=90） | dustbin track2 **三项全改善或不变**（F5cm 0.677→0.763、IoU 0.201→0.227）；同批 track3 dustbin F5cm 0.735→0.535、wardrobe 0.819→0.754 明显变差。聚合 F5cm **−0.028**、voxel IoU **−0.011** |
| per-instance（300 帧连续） | 全部变差。accepted 266 帧但只剩 **2 个** matched object，F5cm −0.017、ghost +0.010 |
| online object pose loop（100 帧） | ATE 0.1192 → 0.1244、RPE 变差、地图 F5cm 0.348 → 0.338，**全项变差** |
| GT feedback POC（50 帧） | **机制验证成功**：已知正确的修正写回累积器后，t+1..t+10 的 translation **10/10 改善**。证明累积器能接受并传播修正，但不证明 SAM 能估出它 |

### 2.2 相机位姿闭环反馈（本轮主线）

| 实验 | 结果 |
|---|---|
| object-consensus feedback（100 帧，baseline） | 主方法 direct ATE +5.3% 但 **sim3 −0.6%**；accepted 帧只有 **38.5%** 局部变好、中位增益 **−2.3 mm** |
| branch `fresh`（有界参考年龄） | anchor 年龄 37.5 → **8.0**、相关性 0.79 → **0.17**（p=0.09），但 `prop_err` **0.0868 → 0.0855 纹丝不动** |
| branch `factorized`（先旋转后平移） | `d_ATE` **−0.0131**（比 raw 更差）、`single` 变体局部变好率 0.667 → 0.143 |
| mask oracle（GT mask 替换 SAM） | `prop_err` **0.1023 vs SAM 0.0845**——**没有改善，反而略差** |

### 2.3 关键的可复现性事实

- **单物体效应可复现**：三次独立 run 中，`dustbin` track2 的 F5cm 提升都是 +0.086
  （0.763275 / 0.763275 / 0.763452），track3 dustbin 的下降都是 −0.20。
- **聚合指标不可复现**：**同一配置**重跑，`d_ATE` 从 0.053 摆到 0.079（+50%），
  sim3 增益从 −0.006 翻到 +0.020。
- 分割模型跨 run 散布 **0.026**，而 `fresh` 相对 baseline 的效应只有 **0.006**——
  **噪声是效应的 4 倍**。
- 分析链噪声约 **1e-2 相对**：vendored 的 `online_motion_averaging` 逐位不可复现
  （从第一个累积帧起差约 1 个 float32 ulp，单线程/零初始化/确定性算法都无效）。

## 3. 为什么失败

### 3.1 约束太弱（根因）

提案的 GT 平移误差中位 **0.086 m**，而它要修正的 raw ATE 只有 **0.117 m**。
**约束携带的信息量和误差本身同量级**——一个 8.6 cm 错的修正去修 11.7 cm 的误差，
平均只能赚一点点。所有下游的加权、共识、门控都只是在噪声里做选择。

### 3.2 物体几何欠约束 6DoF

`geometry_type` 分布：planar 133 + linear 89，**volumetric 只有 5**（共 227）。
平面可以沿自身滑动而 loss 不变——**存在平坦方向**。旋转是最差的一半：共识旋转
误差 **0.611°**，而 raw 逐帧 RPE rotation 只有 **0.318°**——旋转估计比噪声底还差
2 倍。

### 3.3 瓶颈不在 SAM（oracle 结论）

GT mask（完美身份 + 完美边界）**没有降低提案误差**（0.102 vs 0.085），且：

- `track_length` 类内 ρ 从 +0.776 升到 **+0.817**——换 mask 也动不了它
- `inlier_ratio` / `overlap_count` 的预测力**消失**（p 从 0.0002 掉到 0.43 / 0.93）
  → SAM 下那个"匹配越密越准"的相关性，是关于 mask 边界质量的，**不是对应正确性**
- volumetric 样本从 5 涨到 13，误差 **0.115 比 planar 的 0.095 还差** → 几何退化也不是主因

### 3.4 参考年龄是混淆，不是因果

归因曾显示 anchor 年龄与提案误差类内 ρ = **+0.785**（比 track_length 还高），
看起来是主因。但 `fresh` 分支把它压到 0.17 之后，**提案误差没变**。
anchor 年龄只是与 `frame` / `track_length` 共线的产物。

### 3.5 唯一活下来的变量：`track_length`

类内 ρ **+0.78**（baseline）、**+0.82**（oracle），五个类别单独看全为正
（0.59–0.99）。它不随参考新鲜度改变、不随 mask 来源改变、不随几何类型改变。
它与 `delta_translation_norm`（类内 +0.43）一起指向同一件事：

> **随时间累积的漂移让需要估计的修正变大，而物体的退化几何无法精确确定一个大修正。**

### 3.6 目标函数优化的是自洽性，不是正确性

- V3：loss 降 **58%**，GT 指标平坦
- `alignment_loss_after` 类内 ρ = **+0.331**（边际是 −0.197）——**符号翻转**，
  即类别内"loss 更低"反而"提案更差"

原因：优化的是"当前预测点云 ↔ 历史预测点云"。两边带同样的系统误差时，把位姿弯一弯
就能让 loss 降下来，而 GT 距离不变。**修正被要求同时修漂移和修物体自身的深度/mask
误差，而后者不可能靠对齐历史修好，只能被位姿吸收。**

### 3.7 两个"可靠性分数"是反预测的

| 分数 | 边际 ρ | **类内 ρ** |
|---|---:|---:|
| `semantic_confidence`（S_sem 的代理） | +0.414 | **+0.634** |
| `geometry_confidence`（S_geo 的代理） | +0.143 | **+0.342** |

`reliable` 硬拒筛选**零区分度**：保留 0.0881 vs 丢弃 0.0810。

### 3.8 真正有效的是共识与门控

| 筛选 | 保留 | 丢弃 |
|---|---:|---:|
| `consensus_inlier` | **0.0753** | **0.1573** |
| 帧级 gate（accepted vs rejected） | **0.0405** | **0.0914** |

跨物体一致性携带真实信号；**逐物体的可靠性分数没有**。

## 4. 结论

1. **修正机制本身有效，而且单物体层面可复现**。bed 在 shared 模式下四项指标全改善、
   F5cm +32%；dustbin track2 在三次独立 run 中稳定 +0.086。
2. **失败的不是方法，是选择器**。同一场景、同一 pipeline，`dustbin` track2 赢 +0.086、
   track3 输 −0.20。**没有可用的判据区分"这个物体会不会赢"**——两个设计出来的
   分数都是反预测的。
3. **闭环那一半是好的**：累积器确实能接受并传播修正（GT POC 10/10）。坏的从来不是
   loop，是喂进去的 `ΔT`。
4. **不能再声称的**：SAM3.1 能稳定优化 HorizonStream 位姿或稳定提升所有物体点云质量。
   **可以声称的**：persistent object masks 提供了有用但不稳定的局部几何约束；
   self-consistency loss 不能可靠地转化为 GT reconstruction gain。
5. **测量能力的边界**：姿态指标的可复现上限约 **1e-2 相对**（分析链）+ **0.026**
   （分割模型跨 run）。**小于这两个数的分支差异不是结果。**

## 5. 若要继续，唯一有依据的方向

不是再加一个修正分支，而是**造一个能区分"哪个物体会赢"的判据**。

- `track_length` 是唯一活下来的预测因子，但它只在**事后**可得且方向为负（track 越老
  越差）——需要把它变成一个**事前**判据，或者用近期观测密度 / 短窗口重锚来替代
- oracle 已排除 SAM 与几何退化；`inlier_ratio` 在小物体上仍显著（dustbin 类内
  −0.536），这是唯一还活着的正面线索
- 若造不出这样的判据，这条线应写成封闭的负结果：**机制有、选择器没有**

## 6. 复现入口

```bash
# 四个提案侧分支 + mask oracle + 噪声底，一次跑完
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt
```

产出：`<base>.branches.json`、各分支的 `object_pose_feedback/summary.json` 与
`attribution.json`、以及对照表（含两个噪声底）。
