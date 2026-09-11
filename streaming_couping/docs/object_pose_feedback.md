# SAM 物体共识位姿反馈实验（object-consensus pose feedback）

更新时间：2026-09-10（实现完成，服务器实验待跑）

## 1. 实验问题

验证：**SAM3.1 的 persistent object tracks 是否可以作为跨帧稳定的 object anchors，
帮助 HorizonStream 检测并修正累计 pose drift，并把修正反馈到后续帧。**

核心思想：

> Persistent objects do not directly provide the correct camera pose. Each
> object proposes a pose residual, while semantic reliability, geometric
> observability, and cross-object agreement determine whether the residual
> should be fed back into the streaming pose accumulator.

不训练任何模型；不修改 HorizonStream backbone / KV / GLA cache；不引入 DINO 或
learned score。

## 2. 数据流（四段式命令）

```text
zsh streaming_couping/commands_run_scannet_object_pose_feedback_100f.txt
```

```text
Stage 1  horizonstream env (cuda:0)
  generate_horizonstream_geometry_cache --save-chunk-cam-maps
  → horizonstream_geometry.pt + horizonizonstream_geometry_chunk_cam_maps.pt

Stage 2a horizonstream env (SAM cuda:2, 优化 CPU)
  run_semantic_map --object-pose-loss-* --object-pose-loss-independent-instances
                   --object-pose-loss-export-feedback-diagnostics
  → raw_pose/ + object_pose_object_only_per_instance/   （Branch D 地图）
  → object_pose_refinement/feedback_diagnostics.pt      （观测/参考云/提案）

Stage 2b 纯 CPU（可单独廉价重跑调阈值）
  run_object_pose_feedback
  → A raw 重放（并断言 ≡ cache 轨迹）
  → B 全部 proposals vs ΔT_GT 逐个评误差
  → C 五个 consensus 变体各自 gating → 注入重放 → 轨迹指标
  → object_pose_feedback/{frame_metrics,object_proposals,consensus_metrics,
     future_pose_gain}.csv + poses.pt + summary.json

Stage 3  evaluate_exported_semantic_map --map-source tracks
  → Branch D 的 raw vs per-instance 物体点云指标

Stage 4  print_object_pose_loss_object_only_metrics + 反馈 summary
```

默认场景 `00a231a370`、帧窗 `90–189`、prompts `bed wardrobe chair rug dustbin`。
两遍式设计：proposals 全部基于 raw 几何计算；SAM 和 refiner 只跑一次，之后所有
consensus/gating/阈值迭代只重跑 Stage 2b。

## 3. 位姿约定（与已验证的 GT feedback POC 完全一致）

| 对象 | 约定 |
|---|---|
| 模型 chunk 输出 / 内部累加器 | w2c，窗口锚定最新帧；累积为左乘 `inv(rel)` 取中值 |
| 公开轨迹 / 评测 | c2w，frame-0 gauge |
| 物体提案 ΔT_k | c2w 左乘：`C[k,t] = T_aligned[k,t] @ inv(T_raw[t])`，同时也是世界点左乘修正 |
| GT 修正 | `ΔT_GT_t = G_t @ inv(R_t)`（同一 gauge） |
| 注入 | 绝对目标 `target_t = ΔT_consensus @ R_t`，经 POC 原语 `_replace_internal_pose_for_public_target` 写入 `online_absolute_poses` 与 `last_absolute_poses` |
| GT 来源 | manifest `world_to_camera` 求逆 → 归一到首个选中帧；**仅在所有反馈决策冻结之后加载** |

注入语义是**绝对目标**而不是增量：anchor（前 5 帧）钉在 raw 世界系上，因此
重复注入是覆盖，不会叠加过修正。`sliding=1` 时注入粒度为逐帧（仅首 chunk 内的
帧以 chunk 边界为粒度）。

## 4. 可靠性、共识与门控

每个 (frame, instance) 提案先算两个规则化置信度（无 learned score）：

- `S_sem`：track_length / visibility 连续率 / mask 像素 / SAM score 加权归一；
  track 过短、可见率过低、score 过低 → 硬拒（`low_track_confidence`）；
- `S_geo`：alignment 改善 / inlier ratio / overlap / 修正幅度 sanity /
  **协方差特征值退化分类**（λ1≥λ2≥λ3 → volumetric/planar/linear/degenerate，
  退化只降权不直接拒）→ 硬拒原因 `degenerate_geometry` /
  `correction_too_large` / `low_geometry_confidence`。

跨物体共识把每个 ΔT_k 转成 `ξ_k = (rotvec, translation)`（与 refiner 优化变量
同参数化的小角度近似），做加权 Huber IRLS；输出 per-object
`translation/rotation_consensus_error` 与 `consensus_inlier`。

五个 ablation 变体一次运行全部算完（重放是 CPU）：

| 变体 | 聚合 | 权重 |
|---|---|---|
| `single` | 最高权重单物体 | S_sem×S_geo |
| `mean` | 加权平均 | 均匀 |
| `robust` | Huber IRLS | 均匀 |
| `robust_semantic` | Huber IRLS | S_sem |
| `robust_semantic_geometric`（主方法） | Huber IRLS | S_sem×S_geo |

帧级门控按优先级输出 reject reason：
`insufficient_objects → low_track_confidence → degenerate_geometry /
correction_too_large / low_geometry_confidence → no_consensus →
correction_too_large（共识幅度）→ no_alignment_improvement`。
最后一个门用 **pairing snapshot 里的真实参考云** 复算加权 NN 残差：consensus
修正统一作用到所有贡献物体的当前世界点后，残差必须严格下降，而不是用
per-instance 最优 loss 代替。所有阈值集中在 `ObjectPoseFeedbackConfig`
（约 40 个参数，全部可通过命令行覆盖）。

## 5. 分支

| 分支 | 内容 |
|---|---|
| A raw | 无注入重放；内置等价性断言（vs cache 轨迹，平移 <1e-3 m / 旋转 <0.01°） |
| B no-feedback | 只分析 proposals（含被拒 best_pose）与 ΔT_GT 的误差及 clamp 饱和标志 |
| C feedback | 五个变体；主方法 `robust_semantic_geometric` |
| D per-instance | Stage 2a/3 的逐实例物体点修正地图（相机位姿保持 raw） |

## 6. 输出

`object_pose_feedback/` 下：

| 文件 | 内容 |
|---|---|
| `object_proposals.csv` | 每提案一行：track/点数/overlap、loss before/after、inlier ratio、特征值与几何类型、ΔT 幅度、S_sem/S_geo、refiner 与 object 级 reject 原因、consensus 误差与 inlier、GT 修正误差与饱和标志 |
| `frame_metrics.csv` | 每 (variant, frame)：raw/feedback 平移与旋转误差、proposal/reliable 数、accept 与 reject 原因、共识幅度 |
| `consensus_metrics.csv` | 每 (variant, frame)：可靠性计数、inlier 数、聚合损失 before/after、GT 共识误差 |
| `future_pose_gain.csv` | 每 (variant, accepted 帧, k=1..10)：raw/feedback 未来误差与增益 |
| `poses.pt` | raw / gt / ΔT_GT / 各变体轨迹 |
| `summary.json` | 8 个问题的数值答案、bottleneck 归因、GO/NO_GO 决策与判据、`refiner_settings_audit` |
| `attribution.json` | 归因脚本的输出（不参与建图，只在分析时生成） |

### 离线归因

`summary.json` 只回答"整体有没有改善"，不回答"为什么"。当 GO/NO_GO 落在阈值附近时，
用它判断下一步改哪里：

```bash
python -m streaming_couping.scripts.analyze_object_pose_feedback_attribution \
  --feedback-dir outputs/<run>/object_pose_feedback
```

它输出五节，全部离线、不重跑任何模型：

- **(a) 共识塌缩**：在每个变体真正形成共识的帧上统计 `inlier_count`。如果主方法的平均
  inlier 数掉到 1 而 `robust_semantic` 保持 ≥2，说明乘法退化抑制把跨物体共识退化成
  了单物体估计，问题在 S_geo 的用法而非共识本身。
- **(b) 特征预测力**：每个提案特征对 `gt_translation_correction_error` 的 Spearman 秩相关
  与置换 p 值，外加中位数二分表。**一个不能给提案正确性排序的分数，无论怎么加权都不会
  改善共识**——这是判断 S_sem/S_geo 值不值得留的依据。
- **(b2)/(b3)**：按 `geometry_type` 和 `category` 分桶的 GT 误差。
- **(c) 筛选质量**：提案层（refiner 判决、`consensus_inlier`）和被拒帧 vs 接受帧的 GT 共识
  误差。后半部分需要拒绝帧也记录共识 delta（当前 revision 已支持）；旧 CSV 那几格为空时
  脚本会明确提示重跑 Stage 2b，而不是给出误导性的 0。
- **(d) 逐帧局部效果**：接受帧上注入后的位姿是否真的比 raw 更接近 GT（传播之前）。

所有 p 值都是置换检验（默认 4000 次），不是正态近似；提案数只有几百，**不要只看
rho 的大小，要看 p 值和分桶后的样本数**。

RPE 额外报告**排除 correction-boundary 对**的版本（相邻对任一端是修正帧则排除），
避免注入造成的相邻帧 RPE 假跳变。

该边界排除版 RPE 的 translation/rotation 两个 key 由
`object_pose_feedback.RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY` /
`RPE_ROTATION_BOUNDARY_EXCLUDED_KEY` 单点定义，写入方（Stage 2b 重放）和读取方
（`decide_object_feedback`）共用同一常量，避免两端的 key 名漂移。rotation 比值只在
`decisions[*].rpe_rotation_ratio_boundary_excluded_informational` 里作审计输出，
**不进入 `criteria`**，因此不影响 GO/NO_GO。

## 7. GO / NO-GO 判据（仅位姿指标，alignment loss 不进判据）

主方法 `robust_semantic_geometric` 需同时满足：

1. direct ATE 相对 raw 改善 ≥ 5%；
2. accepted 帧的 future translation gain（t+1..t+10）中位数 > 0 且 ≥ 60% 为正；
3. 排边界 RPE translation 不劣于 raw × 1.05；
4. accepted ratio（有提案的在线帧中）≥ 10%；
5. **sim3 ATE 不劣于 raw**（`decision_min_sim3_improvement_ratio`，默认 0）。
   direct ATE 单独不够——修正可以只吸收一个全局相似变换（尺度/刚体）偏移而相对
   轨迹完全没变。两种失败模式都实测到了：主变体 direct +5.3% 而 sim3 −0.6%；
   **同一配置**重跑一次，sim3 增益从 −0.006 翻到 +0.020。
6. **accepted 帧的 future rotation gain 中位数 ≥ 0**
   （`decision_min_future_rotation_gain_deg`，默认 0）。旋转修正比它要修的轨迹噪声
   底还差（共识旋转误差 0.61° vs raw RPE rotation 0.32°），baseline 的 accepted
   帧旋转增益中位数是负的。这条防止"只帮了平移"的修正被当成位姿改善。

两条守卫的阈值都可以通过 `ObjectPoseFeedbackConfig` 放宽（设为负值即等于关闭），
但默认值是数据支持的。

不满足则 `OBJECT_FEEDBACK_NO_GO`，summary 同时给出启发式瓶颈归因
（`sam_tracking` / `nn_correspondence_or_object_geometry` / `consensus` /
`gating` / `pose_feedback`）。

## 8. 已知边界

- 提案受 refiner clamp 限制（≤10°/0.25m）；`|ΔT_GT|` 超出时用饱和标志区分
  "提案不准" 与 "提案被 clamp"。
- 运动平均的中值会稀释单帧修正（窗口内 9 个候选取中值）；持续多帧注入同一
  方向修正才能稳定传播，这是 accumulator 机制本身的属性，不是 bug。
- 两遍式：proposals 基于raw 几何一次性计算，修正后的世界不回灌给后续提案
  （对 300 帧实验暴露的 reference 污染问题更鲁棒；交织式留作后续实验）。
- `anchor_frame_count` 等反馈配置需与 Stage 2a 的 refiner 参数保持一致（默认 5）。
  该约束现在**强制校验**：Stage 2a 会把 refiner 的 `anchor_frame_count`、
  `max_correction_{rotation_deg,translation_m}`、`max_match_distance_m`、
  `trim_ratio`、`min_matches_per_pair` 写入 `feedback_diagnostics.pt` 的
  `refiner_settings`，Stage 2b 启动时与反馈配置逐项比对（见
  `object_pose_feedback.REFINER_SETTING_MIRRORS`）；不一致直接报错并列出差异，
  结果记录在 `summary.json` 的 `refiner_config_match` /
  `refiner_settings_audit`。确实想用非镜像阈值时传
  `--allow-refiner-mismatch`，降级为 warning 并把该 run 标记为
  `refiner_config_match=false`。
- GT 只在 Stage 2b 所有决策之后加载，summary 记录 `gt_used_for_feedback: false`
  审计标志。

## 8.5 提案侧实验分支（参考新鲜度 / factorized 6DoF）

100 帧基线的归因把瓶颈定位到了**提案本身**，而不是下游的聚合与验证：

- 提案的 GT 平移误差中位数 **0.086 m**，raw ATE 只有 **0.117 m** —— 同一量级；
  下游的 consensus 和门控只是在噪声里做选择。
- 误差的主导变量是**参考集的年龄**。`anchor_reference_age`（帧 0–4 的固定 anchor，
  中位 37.5 帧）与 GT 修正误差的相关 ρ = **+0.933**，**类内 +0.785** —— 比
  `track_length`（+0.776）还高，说明 `track_length` 一直只是 anchor 年龄的代理。
- 而 local history（中位 **1 帧**）的 ρ = −0.088，**p = 0.32，完全无信号**。
- refiner 给前者权重 **1.0**、后者 **0.50**：**决定误差的那一半拿满权重，零信号的
  那一半被砍半。**
- 两个"可靠性分数"也都是反向的（S_sem 类内 +0.634、S_geo 类内 +0.342），
  `reliable` 硬拒筛选零区分度（0.0866 vs 0.0859）。真正有效的是 consensus inlier
  测试（0.0753 vs 0.1573）和帧门控（0.0405 vs 0.0914）。

因此这一轮改的是**提案**，不是聚合：

| 配置项 | 作用 | 默认 |
|---|---|---|
| `max_reference_age_frames` | 丢弃比当前帧老这么多帧的参考 | `0`（不限制 = 原行为） |
| `anchor_refresh_interval_frames` | 每 N 帧重新采集高权重的 anchor 集，而不是钉在序列开头 | `0`（不刷新 = 原行为） |
| `proposal_mode` | `joint`（原行为）或 `rotation_then_translation` | `joint` |

factorized 的依据：共识 rotation 误差中位 **0.611°**，而 raw 逐帧 RPE rotation
只有 **0.318°** —— 旋转估计误差是噪声底的 2 倍，且本场景物体 97% 是 planar/linear
（volumetric 只有 5 个样本），平移和旋转在联合优化里会互相补偿。

**一键跑全部四个分支**：

```bash
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt
```

**只有 baseline 跑完整 pipeline（唯一的 GPU 阶段）；其余三个分支用
`run_object_pose_loss_replay` 从 baseline 缓存的观测重放 refiner，纯 CPU。**
对比表在最后打印并写出 `<base>.branches.json`。每个分支写进自己的
`<base>.<branch>` 目录，**不碰你已有的结果**，所以 `<base>.baseline` 与现有 run
的对比就是"默认值有没有改变旧行为"的回归检查。

为什么不是四个分支各跑一遍 pipeline：旋钮在 stage 2a 里、紧挨着分割模型，而分割
模型在 GPU 上跨 run 不确定。上一轮这样跑的后果是——**同一配置**的 `d_ATE` 从
0.053 摆到 0.079、`d_sim3` 翻转符号，而分支效应只有 0.006：**噪声是效应的 4 倍**。
重放路径没有 GPU、也没有分割模型的方差，分支差异因此只来自被测的那个旋钮。

重放前有一道**闸门**，而且它检查的是**整条链**而不只是 proposals：用 baseline 自己的
设置重放 → 重跑 Stage 2b → 用 `compare_json` 比对两边 `summary.json` 的 `decisions`。
任何实质差异都会中止，因为那说明分支差异建立在一个不稳定的链条上。这道检查是免费的
（纯 CPU），它测的是分析链的确定性。

### 两个噪声底

**分析链噪声（免费、每次运行都测）**：闸门把 baseline 的 `summary.json` 和"重放同一份
缓存后重跑 Stage 2b"的 `summary.json` 逐叶子比对，并**区分**：

- **结构性变化**（某个 `passed` 翻转、`decision` 改变、字段出现/消失）→ **中止**，链条坏了
- **数值漂移**（浮点）→ 报出最大相对漂移，**继续**

实测：同一份缓存输入重放两次，姿态指标漂移 **1.0e-2** 相对量级。根因是 vendored 的
`online_motion_averaging` 逐位不可复现——从第一个累积帧开始差约 1 个 float32 ulp，
单线程、零初始化 buffer、`use_deterministic_algorithms` 都消不掉。这个 1e-7 的轨迹差
被"接近零角的旋转测量"放大成 1e-2。

**分割模型噪声（`SAM_REPEATS`，默认 `3`）**：控制**完整 baseline pipeline** 跑几次，
测出分割模型跨 run 的散布，打印在对比表上方：

```
noise floor from 3 runs of one configuration
  direct_ate_improvement_ratio  spread=0.0xx (relative 0.xxx)
```

**任何一个噪声底都大于分支效应时，那张表读不出结论。** 上一轮分割模型的实测散布是
0.026（相对 ~0.33），而 `fresh` 的效应只有 0.006——**噪声是效应的 4 倍**。分析链噪声
（1e-2）同样不可忽略。

**根因说明**：vendored 的 `online_motion_averaging` 逐位不可复现（从第一个累积帧
起差约 1 个 float32 ulp），单线程 / 零初始化 buffer / `use_deterministic_algorithms`
都无效。所以姿态指标的可复现上限大约是 1e-2 相对量级——**小于这个的分支差异，
无论看起来多合理，都不是结果**。要更小的噪声必须修 HorizonStream 内部或在
torch 之外重写累积器。

第二轮的实测结果（见 §8.6）：参考年龄假设被否证，噪声底大于效应。所以这个表目前
的用途是**量化噪声**并确认机制是否生效，不是拿来选分支。

### 重放 refiner（`run_object_pose_loss_replay`）

```bash
python -m streaming_couping.scripts.run_object_pose_loss_replay \
  --diagnostics <run>/object_pose_refinement/feedback_diagnostics.pt \
  --output-dir  <branch-dir> \
  --object-pose-loss-max-reference-age-frames 15 \
  --object-pose-loss-proposal-mode rotation_then_translation
```

它只读 `feedback_diagnostics.pt`（观测、帧 id、raw poses、**完整** refiner 配置、
预过滤计数都在里面），写出 `<output-dir>/object_pose_refinement/feedback_diagnostics.pt`
——正是 stage 2b 期望的位置，所以后续分析原样可用。`--check-equivalence` 会用源配置
重放并断言 proposals 一致。

单个分支也可以单独跑，由 `commands_run_scannet_object_pose_feedback_100f.txt`
顶部的 `PROPOSAL_BRANCH`（或 `OBJECT_POSE_FEEDBACK_PROPOSAL_BRANCH`）选择：

| 分支 | 内容 | run 目录后缀 |
|---|---|---|
| `baseline` | joint + 固定 anchor（等于改动前行为，逐位一致） | 无 |
| `fresh` | joint + 有界参考年龄（age 15 / refresh 20） | `.fresh` |
| `factorized` | rotation→translation + 固定 anchor | `.factorized` |
| `fresh_factor` | 两者叠加 | `.fresh_factor` |

**注意**：三个默认值合起来必须逐位复现改动前的行为，否则 baseline 对照失效。
Stage 1 的几何 cache 与分支无关，四个分支共用同一份（`--reuse-if-valid`），
所以只有第一个分支付几何推理的 GPU 成本。

已知风险：丢弃 anchor 会让每帧的参考对变少（history 上限只有 2），可能增加
`too_few_initial_object_matches` 的拒绝数、减少提案总量。所以新分支要先看
`proposal_count` 有没有明显下降，再比较提案误差。

## 8.6 第二轮分支 sweep 的结果（参考年龄假设被否证 + 噪声底）

四分支 sweep 跑完后（`<base>.branches.json`）：

| | baseline | fresh | factorized | fresh_factor |
|---|---:|---:|---:|---:|
| `anchor_age` 中位（帧） | 37.5 | **8.0** | 37.5 | **8.0** |
| `anchor_rho` 类内 | 0.791 | **0.166** | 0.760 | **0.161** |
| **`prop_err` 中位（m）** | 0.0868 | **0.0855** | 0.0853 | 0.0927 |
| `proposals` | 229 | 182 | 229 | 182 |
| `d_ATE` | +0.0790 | +0.0731 | **−0.0131** | +0.0235 |
| `d_sim3` | +0.0195 | **−0.0137** | −0.0138 | −0.0116 |
| decision | GO | GO | NO_GO | NO_GO |

**结论一：参考年龄是混淆，不是因果。** `fresh` 机械上完全生效——anchor 年龄从
37.5 压到 8.0，类内相关性从 0.79 掉到 0.17（不再显著）——但**提案误差纹丝不动**
（0.0868 → 0.0855）。之前那个 ρ=+0.785 只是 anchor 年龄与 `frame` / `track_length`
共线的产物；真正驱动误差的东西不随参考新鲜度改变。残留的最强预测因子仍是
`track_length`（fresh 分支里类内 ρ 反而升到 0.809）和 `semantic_confidence`（0.726）。

**结论二：factorization 让提案更差**（超出噪声）。`d_ATE` 转负、(d) 里 `single` 的
局部变好率从 0.667 掉到 0.143。

**结论三（最重要）：这个测量目前的噪声大于要测的效应。** baseline 分支跑的是
改动前的默认行为，可以和你原来的 run 直接对比：

| | 原 run | baseline 分支 | 变化 |
|---|---:|---:|---:|
| `d_ATE` | 0.0528 | **0.0790** | **+50%** |
| `d_sim3` | −0.0057 | **+0.0195** | **符号翻转** |
| accepted（主变体） | 13 | 16 | +23% |
| `prop_err` | 0.0856 | 0.0868 | +1.4% |

同一份代码、同一场景、同一帧窗：`d_ATE` 摆动 **0.026**，而 `fresh` 相对 baseline 的
效应只有 **0.0059**——**噪声是效应的 4 倍**。所以 `fresh` 那一行没有判别力，且此前
所有落在 5% 阈值附近的 GO/NO_GO 都不可靠。

对比之下 `prop_err` 跨 run 只差 **1.4%**，是稳定得多的端点。

**结论四：旋转角的测量公式此前有数值地板。** `acos((trace-1)/2)` 在接近单位阵时
病态：一个只在 float32 精度上正交的旋转矩阵（trace 2.999852）**与自己做比较**也会
报出约 **0.028°**，因为 acos 以 sqrt(2ε) 放大误差。已改为 `atan2(sin, cos)` 形式
（`rotation_angle_deg`）。受影响的是所有用该公式的地方：`proposal_rotation_median_deg`
（1.34°，约 2% 偏差）、共识旋转误差（0.611°，约 5%），以及**主变体的
`future_rotation_gain_median_deg`（−0.0105°）和 `single` 的（−0.0130°）——这两个
本来就低于旧地板，属于噪声**。轨迹 RPE 走 POC 的 SVD 投影路径，不受影响。

## 8.7 Mask oracle 对照（GT mask 替换 SAM mask）

动机失败在一个结构性缺口上：SAM 给的是 **2D 支撑 + 身份**，要变成位姿约束还需要
(i) 跨帧对应正确、(ii) 物体三维结构非退化。两者都有量化失败证据——但**无法区分**是
分割模型的问题还是对应/几何的问题。唯一能拆开的实验是把 mask 换成 GT。

```bash
python -m streaming_couping.scripts.run_object_pose_loss_oracle \
  --geometry-cache <run>/horizonstream_geometry.pt \
  --manifest <manifest> --scene-id 00a231a370 \
  --prompts bed wardrobe chair rug dustbin \
  --output-dir <base>.oracle_mask
```

**它只需要 CPU**：几何来自缓存的 HorizonStream cache，mask 来自 manifest 里的
`instance_mask` 标注，完全绕过分割模型和 HorizonStream。

读法——只看 `prop_err`，和 baseline（SAM）比：

| 结果 | 结论 |
|---|---|
| oracle 的 `prop_err` 明显更小 | 瓶颈在**分割模型**：身份、mask 边界、或 re-entry |
| oracle 的 `prop_err` 几乎不变 | 瓶颈在 mask **下游**：最近点对应，或物体几何退化 |

**它是诊断上界，不是方法分支。** 它故意违反"GT 不进入候选生成"这条项目规则——这正是
对照的意义。产出的 `feedback_diagnostics.pt` 里带 `oracle` 审计块
（`gt_used_for_proposals: true`、`purpose: diagnostic_upper_bound_only`、
`not_a_method_result: true`、`segmentation_model_bypassed: true`），目录名是
`<base>.oracle_mask`，**任何情况下都不能当作系统的一个分支报告**。

sweep 命令文件会自动先跑这个对照（CPU、几秒，而且排在 GPU 之前，所以即使后面的
pipeline 失败它的结果仍然保留），并把它作为第五行放进对比表。

## 9. 相关代码

| 文件 | 角色 |
|---|---|
| `src/semantic_mapping/object_pose_feedback.py` | 配置、S_sem/S_geo、退化分类、consensus、gating、聚合损失、决策与归因 |
| `scripts/run_object_pose_feedback.py` | Stage 2b 入口 + 重放引擎（复用 POC 注入原语） |
| `object_pose_loss_refinement.py` | 新增 `export_feedback_diagnostics`（默认关）：observations / pairing_snapshots / rejected best_pose |
| `generate_horizonstream_geometry_cache.py` | 新增 `--save-chunk-cam-maps`（默认关）：chunk 相机图 aux 文件 |
| `scripts/run_scannet_horizonstream_gt_feedback_poc.py` | 已验证的注入原语与评测函数（本实验原样复用，未修改） |
| `scripts/analyze_object_pose_feedback_attribution.py` | 离线归因：共识是否塌缩成单物体、可靠性分数是否真能预测提案对错、筛选有没有选对、参考年龄 |
| `scripts/run_object_pose_loss_replay.py` | 从 `feedback_diagnostics.pt` 重放 refiner（纯 CPU、确定性），使分支对比不含分割模型的方差 |
| `scripts/run_object_pose_loss_oracle.py` | mask oracle 对照：用 GT instance mask 跑 refiner（纯 CPU），拆分"分割模型"与"对应/几何"两个瓶颈 |
| `tests/test_object_pose_loss_oracle.py` | oracle 注入的是 GT 身份与标注，且带不可当方法结果的审计标志 |
| `scripts/summarize_feedback_branches.py` | 多分支横向对比表 |
| `tests/test_object_pose_loss_replay.py` | 重放忠实性：同一份观测重放必须复现源 proposals |
| `tests/test_object_pose_feedback.py` | 20 个 CPU 测试：Lie 代数、退化分类、可靠性规则、共识拒外点、门控全部 reject 原因、**重放等价性**、注入语义、RPE key 契约、refiner 阈值镜像校验、拒绝帧仍携带共识 delta |
| `tests/test_analyze_object_pose_feedback_attribution.py` | 9 个 CPU 测试：塌缩检测、信号/噪声特征区分、筛选质量、缺列时显式报错 |

注意 `test_replay_equivalence_with_motion_averaging` 和 `test_replay_injection_semantics`
需要 `horizonstream` 可导入，否则 pytest 会 skip。本地/CI 运行时要显式加上子模块路径：

```bash
PYTHONPATH="$PWD:$PWD/externals/horizonstream" \
  python -m pytest streaming_couping/tests/test_object_pose_feedback.py -q
```
