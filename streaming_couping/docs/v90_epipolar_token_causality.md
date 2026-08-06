# V9：SAM3.1 local-token 二维极几何位姿因果实验

> 状态：`Stage O-R1 relative oracle 与 A0-H best-condition 已全折通过；Stage A0 all-edge 和 A1 SAM matcher 已完成但未通过；absolute trajectory 不作为 token 主命题`
> 日期：2026-08-06
> 核心目标：验证 SAM3.1 local descriptor 是否能通过真实 2D–2D 对应改善 StreamVGGT 的
> 相机 rotation 与 translation direction，而不使用 predicted depth、pointmap Kabsch 或
> learned pose residual。

## 1. 为什么这是新实验

V4/V5 已做 instance token → DPT patch → depth/pointmap；V7.2 已做 SAM local token → direct
pose residual；V7.3/V7.4 已做 SAM affinity → StreamVGGT 3D geometry Value → direct pose
residual；V8 已做 GT-world 3D pseudo-match → predicted depth/Kabsch。

V9 改变的是几何观测本身：

```text
SAM3.1 local token + 2D coordinate
  → current/history 真实像素对应概率
  → 固定 calibrated epipolar/bearing solver
  → rotation + translation direction
  → 使用 L0 edge length 或多历史射线交会恢复 center
```

它明确不做：

- 不预测 depth 或 pointmap residual；
- 不读取 StreamVGGT predicted depth；
- 不传输 StreamVGGT 3D geometry Value；
- 不训练 camera/SE(3) head；
- 不把 camera hidden、frame index 或 sequence embedding输入 matcher；
- 不用 GT pose error 选择是否接受修正；
- 不重新 sweep K、hidden dimension、head 数或 fallback threshold。

因此它不会重复 V4–V8 已失败的三条捷径：DPT latent correction、错误 predicted geometry、
feature-to-pose memorization。

## 2. V9 只回答的命题

在已知相机内参的 calibrated setting 下：

1. SAM3.1 local token 能否比 mask/UV 和 StreamVGGT patch appearance 更准确地匹配同一静态
   实例的跨帧表面点；
2. 冻结匹配后，这些对应能否通过固定二维极几何改善 frozen StreamVGGT L0 的 rotation 和
   translation direction；
3. SAM-off、wrong-ID、shuffle-time 是否同时破坏匹配和 pose；
4. frame 90 后出生的动态实例，从第二次观测开始能否提供相同类型的约束。

本实验不声称恢复新的 metric translation scale。单个相机对的 essential geometry 只能决定
translation direction；center 的尺度由 L0 edge length 保留，或由两个及以上已定位历史相机的
方向射线交会得到。

## 3. 相机内参边界

主实验使用 ScanNet++ 提供的相机标定 K。K 是传感器 calibration，不是 GT pose，但对于无标定
互联网图片仍属于 oracle 条件。原因是 O2.6/O2.7 已证明 predicted K 会独立破坏 pose；在同一
实验中同时更换 K 会再次混淆 SAM token 结论。

predicted K 只增加一个 report-only 行，不参与 V9 是否通过。若主结果通过，再单独研究可部署
K；若主结果失败，不允许用 predicted K 失败解释或继续扩大 matcher。

## 4. 数据与动态实例

复用 V7.4/V8 的同一长序列协议：

```text
scene: 00a231a370
frames: 90:15:525，共 30 帧
SAM3.1: forward-only bed/wardrobe discovery
slots: 最多 8 个，按首次出现永久分配
```

frame 90 只定义相机 gauge，不是物体 reference。每个 slot：

- 第一次出现：写入 2D coordinate、SAM descriptor 和 StreamVGGT patch descriptor；
- 第二次出现：可以和严格更早的同 slot observation 匹配；
- 当前帧求解结束后才写 memory；
- 固定读取最近两次有效历史，不做 history-length sweep；
- 新出生实例与 frame 90 已存在实例使用完全相同的逻辑。

## 5. Stage O：可见性正确的 2D oracle 与 solver 上界

在训练任何 matcher 前，先验证二维路线本身是否具备 pose 上界。

### 5.1 GT 只用于构造训练标签/诊断 oracle

对当前 SAM slot mask 内的每个 query pixel：

1. 读取 mesh-rasterized GT depth/GT world point；
2. 用 history GT pose 和 calibrated K 投影到 history image；
3. 检查投影在图像内、深度为正；
4. 与 history GT depth 做 z-buffer visibility consistency；
5. 投影位置必须落在同一个 SAM persistent slot 的 history mask 内；
6. 保留连续亚像素 `uv_target`，不把最近的离散 token index 当作真实对应本身。

这与 V8 的“同 slot GT-world 3D nearest neighbour”不同。V9 标签代表同一个可见 mesh surface
point 的真实重投影，避免相邻但不是同一表面点的 3D pseudo-match。

GT depth/pose 只用于 label、oracle 和评分；learned matcher 的推理输入不包含这些字段。

### 5.2 固定求解器

每个 current-history pair 使用 normalized camera bearing：

```math
b = K^{-1}[u,v,1]^T / \|K^{-1}[u,v,1]^T\|_2
```

主 residual 是 weighted Sampson/epipolar error。求解器不训练参数：

1. weighted normalized eight-point 初始化 essential matrix；
2. SVD 强制 essential rank-2/equal-singular-value 约束；
3. 四个分解中用 cheirality 与“最接近 L0 rotation/direction”确定唯一解；
4. 固定次数的 robust bearing Gauss–Newton 在 L0 附近精修；
5. 单 history 保留 L0 center edge length，只修 direction；
6. 多 history 将各历史相机中心的 translation direction 做加权射线交会，再与 L0 gauge 对齐。

没有 learned accept/reject gate。唯一不更新条件是数学不可求：有效对应少于 8、非有限输入或
明确退化的 design matrix；这些帧逐元素保持 L0，并单独报告，不能被计作改善。

### 5.3 Oracle 停止条件

`oracle_surface_reprojection` 必须在 short/medium/long 三个 fold 同时满足：

- test rotation error 低于 L0；
- test translation-direction error 低于 L0；
- active frame 中没有 aggregate pose 指标恶化；
- inactive/degenerate frame bit-exact 等于 L0；
- reference/gauge bit-exact。

若 oracle 不通过，V9 立即停止，不训练 SAM matcher。此时结论是二维实例支持或 solver 不足，
不是 SAM token 失败。

Stage O 的核心物理输出是每条 current-history essential edge 的 `relative rotation + relative
translation direction`；center distance 只报告，不参与该 relative pass。首版 runner 额外把
`absolute rotation + reference-to-current center direction` 误作为唯一 `fold_oracle_pass`，从而把
单目不可恢复的 edge scale 与 frozen L0 history-center error 混入停止门。该旧字段和失败结果继续
保留为更强的 `absolute trajectory pass`，不能删除或改写；O-R1 另行输出与原始 V9 命题一致的
`fold_relative_oracle_pass`，relative aggregate 固定为两个角度之和，不做调权。

## 6. Stage A：显式 2D correspondence 监督

Stage O 通过后，训练小型 matcher。SAM3.1、StreamVGGT 和 pose solver 均冻结。

### 6.1 统一 token support

每个 slot/frame 固定采样 32 个位置；所有分支读取相同 query/key UV、valid mask、slot、history
和帧 support。只允许 descriptor 内容不同，防止 active coverage 造成虚假提升。

SAM descriptor 直接读取缓存中与 `sam_local_uv` 同时采样的 32 个 local feature。StreamVGGT appearance control 从
相同位置采样 frozen patch token。UV 不拼入 descriptor projection，防止绝对位置成为 frame code；
它只用于 label 和 control support。

### 6.2 分支

| 分支 | correspondence 信息 | 参数/用途 |
|---|---|---|
| `oracle_surface_reprojection` | GT 可见表面点 | solver 上界，不训练 |
| `mask_uv_uniform` | 同 slot 内 uniform/固定 UV proximity | 无 descriptor 下界 |
| `stream_patch_match` | frozen StreamVGGT 2D patch appearance | generic appearance control |
| `sam_match` | frozen SAM3.1 local descriptor | SAM 主候选 |
| `sam_train_off` | 与 `sam_match` 同模块，训练时 descriptor 恒为零 | parameter-matched control |

`sam_match` 与 `stream_patch_match` 使用相同 projector dimension、层数、OT temperature、训练步数
和随机种子。不得给 SAM 分支额外 hidden capacity。

### 6.3 损失

只训练 descriptor Q/K projector，不训练 pose：

```text
visibility-aware soft target q_ij
predicted row-softmax probability p_ij + dustbin
L_match = cross_entropy(q, p)
        + cycle-consistency
        + dustbin/non-visible supervision
```

soft target 由离散 history token 到连续 `uv_target` 的距离生成；超出可见区域或 history mask 的
query 监督到 dustbin，不强迫错误匹配。禁止把 pose loss 接回 matcher。

训练后冻结 matcher，并用 SHA/bit-exact 检查确保 Stage B 不改变任何 Q/K 参数。

### 6.4 Held-out matching 指标

- valid visible query coverage；
- PCK@8px（首次实验锁定，不扫描阈值）；
- mean pixel EPE（拒配 visible query 以图像对角线作有限报告惩罚）；
- soft-target cross-entropy 与 cycle loss；
- fixed solver Sampson RMSE；
- dustbin accuracy；
- descriptor projector gradient norm；
- matching-frozen exact。

## 7. Stage B：固定 matcher → 固定 pose solver

Stage B 不训练任何参数。将各分支的 `p_ij` 直接作为固定 solver 权重，输出：

- relative rotation error；
- translation-direction angular error；
- center error（保留 L0 edge length/多历史射线交会后）；
- secondary full pose loss；
- active/degenerate frame、对应数、design-matrix condition、Sampson RMSE。

主要 pose claim 使用 rotation + translation direction；center 只作为尺度处理后的次要指标。

## 8. 时间 fold

沿用既有动态实例 fold，不新增 sequence split：

| Fold | matcher 训练 | 严格未来测试 |
|---|---|---|
| short | 270–345 | 360–405 |
| medium | 270–405 | 420–465 |
| long | 270–465 | 480–525 |

固定训练步数并保留最后 checkpoint；`best_step` 只作 train log，不选择 checkpoint，更不会根据
test 选择模型。未来帧 target、descriptor 和 memory 不进入训练前缀。所有 fold 使用同一 frozen
L0 和同一 solver 配置。

本实验是同场景时间外推，不是跨场景泛化。

## 9. 因果扰动

只对训练完成的同一个 `sam_match` checkpoint 做：

| 扰动 | 操作 | support 是否变化 |
|---|---|---|
| `sam_off` | descriptor 置零 | 否 |
| `wrong_sam_identity` | 在当前帧循环交换已出生 slot descriptor | 否 |
| `shuffle_sam_time` | 当前 slot 读取另一个过去时间的 descriptor，UV/history 不变 | 否 |
| `channel_permute_fixed` | 固定通道排列，破坏训练投影语义但保持数值分布 | 否 |

扰动必须同时伤害 held-out matching 和固定 solver pose，只有 pose 波动或只有 matching 波动均不
足以证明完整因果链。

## 10. 严格通过条件

`fold_sam_epipolar_pass=1` 必须同时满足：

1. Stage O oracle 在该 fold 通过；
2. `sam_match` 的 EPE 与 PCK 同时优于 `mask_uv_uniform`；
3. `sam_match` 同时优于 parameter-matched `sam_train_off`；
4. `sam_match` 至少不差于 `stream_patch_match`，否则只能证明 generic appearance 有用；
5. 固定 solver 的 rotation 与 translation-direction 同时优于 L0 和三个非 SAM 控制；
6. 四个 SAM 扰动同时使 matching 和 pose 变差；
7. learned 分支的 query/key/active-frame support 完全一致；
8. matcher frozen、reference exact、inactive fallback exact。

`all_folds_sam_epipolar_pass=1` 要求 short/medium/long 全部通过。首次运行不设置人为 1% margin；
只要求预定义指标方向严格一致，并输出逐 frame paired difference。若 seed 0 全 fold 通过，再运行
3 seeds 和按 frame/instance 分组 bootstrap；未通过时不靠改阈值挽救。

## 11. 结果解释树

```text
Oracle 2D correspondence 失败
  → 二维实例支持/epipolar solver 不具备上界；停止，不训练 matcher

Oracle 通过，SAM matching 不超过 controls
  → SAM3.1 detector_fpn2 不提供所需 surface correspondence

SAM matching 通过，固定 pose 不改善
  → 对应误差仍超过 epipolar solver 可用范围，或相机运动/平面结构退化

SAM matching 与 pose 均通过，但 predicted K 行失败
  → 得到 calibrated-camera 下的 SAM-token E3 结论，K 是独立部署问题

三个 fold 全通过
  → 首次得到“正确 SAM local token → 2D correspondence → 固定几何 → pose”的因果证据
```

## 12. 已实现入口与输出

Stage O 只保留一条正式命令；缓存不存在时会先重建 V7.4 的 30 帧动态实例观测缓存，存在时不会
加载 SAM3.1 或 StreamVGGT：

```text
streaming_couping/commands_v90_epipolar_token_causality.txt
```

输出集中到：

```text
outputs/streaming_couping_v90_epipolar_token_causality/
├── v90_oracle_summary.csv             # short/medium/long 各一行，首要复制
├── v90_oracle_frames.csv              # 每个 test frame 的 paired pose 指标
├── v90_correspondence_diagnostics.csv # 每 slot 重投影和每 history edge 的可见性/退化
├── v90_decision.md
└── v90_metadata.json
```

如果 Stage O 失败，仍写出完整 oracle CSV 和 decision，然后正常停止；不创建空 matcher
checkpoint，也不自动进入下一种模型。

### 12.1 Stage O 初次结果（保留，不覆盖）

初次运行使用 eight-point 单初始化、按 effective correspondence 融合 rotation，并对多历史中心做
无限射线交会。结果为：

| fold | rotation gain | direction gain | active worse frames | pass |
|---|---:|---:|---:|---:|
| short | +17.5365% | +27.7961% | 1 | 0 |
| medium | -338.383% | -623.514% | 3 | 0 |
| long | +18.8190% | -230.401% | 2 | 0 |

`all_folds_oracle_pass=0`。所有 test frame 均 active，平均可见对应为 267–569，因此不是 coverage
不足。逐 edge 诊断定位出两类 solver 问题：

- 435/480/510 等帧的 GT surface label 深度残差仍只有毫米到厘米级，但 eight-point essential 的
  Sampson RMSE 达 `0.02–0.43`，平面/低视差 nullspace 选择在 absolute 合成前已经失败；
- 405/420 的 edge Sampson RMSE 约 `4e-4` 且 relative pose 良好，但无限射线交会仍放大 frozen
  L0 history center 的误差。

该结果不涉及 SAM descriptor，因此只能否定初始 solver，不能写成 SAM token 失败。

### 12.2 O-R1 solver 修正结果

O-R1 不改变数据、fold、correspondence 或主指标，仅修正已定位的求解器：

1. 同时从 eight-point decomposition 与 frozen L0 relative pose 初始化固定 Gauss–Newton；
2. 只按同一批 observed correspondence 的 robust Sampson objective 选候选，绝不读取 GT pose
   error，也不增加效果阈值；
3. 输出 `initialization/eight_point_sampson_rmse/l0_local_sampson_rmse`；
4. 每条 edge 仍保留自己的 L0 edge length；多历史 center 改用带 inlier/Sampson 质量的有界凸
   融合，不再做可能发散的无限射线交会；rotation 使用同一质量权重。

为保留初次失败结果，O-R1 默认写入新目录：

```text
outputs/streaming_couping_v90_epipolar_token_causality_solver_corrected/
```

O-R1 已运行。双初始化把初版失败 edge 的 Sampson RMSE 从 `0.02–0.43` 降至通常
`3e-4–8e-4`；12 个 test frame 的 relative aggregate 均未恶化：

| fold | relative rotation gain | relative direction gain | relative worse frames | relative pass |
|---|---:|---:|---:|---:|
| short | +75.6472% | +91.9972% | 0 | 1 |
| medium | +32.9042% | +86.9888% | 0 | 1 |
| long | +78.3661% | +58.3310% | 0 | 1 |

因此 `all_folds_relative_oracle_pass=1`，二维 correspondence → fixed essential solver 的上界成立。
与此同时，short/medium 的 absolute center direction 仍下降，故
`all_folds_absolute_trajectory_pass=0`。这不是继续调 center fusion 的许可：单目 edge 没有尺度，
当前阶段应停止 absolute center 优化，只允许 Stage A 判断 SAM local descriptor 能否逼近 oracle
relative correspondence。即使 Stage A 通过，也只能声明相对 rotation/direction 约束有帮助。

已实现文件：

```text
streaming_couping/src/v90_epipolar_geometry.py
streaming_couping/scripts/run_v90_epipolar_oracle.py
streaming_couping/scripts/smoke_v90_epipolar_oracle.py
streaming_couping/configs/v90_epipolar_oracle.yaml
streaming_couping/tests/test_v90_epipolar_geometry.py
```

当前实现读取 raw `tracking_masks_stream` 的永久 SAM3.1 slot，而不是 geometry-trusted mask；因此
不会重新引入 V6 的 geometry gate。GT world point/depth/global W2C 只存在于 surface-label 函数；
本质矩阵求解和 absolute pose 合成只接收 2D pixels、calibrated K 与 frozen L0 history。唯一 exact
fallback 原因是少于 8 点、非有限、design-matrix/cheirality 退化或线性代数失败，绝不读取 GT
pose error 决定是否接受结果。

命令开头的 smoke 同时覆盖动态 birth、NaN-aware 亚像素 z-buffer、完美对应的 essential pose、
少于 8 点的 exact fallback，以及一个完整的合成逐帧 runner/CSV-schema 路径；服务器不需要
`pytest` 即可先执行这些检查。

### 12.3 Stage A0/A1 实现（已运行）

Stage A 使用独立目录，不覆盖 dense Stage O 初版或 O-R1 结果：

```text
streaming_couping/commands_v90_stage_a_correspondence.txt
streaming_couping/configs/v90_local_token_matcher.yaml
streaming_couping/src/v90_explicit_matcher.py
streaming_couping/scripts/run_v90_local_token_matcher.py
streaming_couping/scripts/smoke_v90_local_token_matcher.py
streaming_couping/tests/test_v90_explicit_matcher.py

outputs/streaming_couping_v90_epipolar_token_stage_a/
├── v90_stage_a_summary.csv
├── v90_stage_a_frames.csv
├── v90_stage_a_training.csv
├── v90_stage_a_pairs.csv
├── v90_stage_a_edges.csv
├── v90_stage_a_problem_edges.csv
├── v90_stage_a_decision.md
├── v90_stage_a_metadata.json
└── checkpoints/
```

一键命令的固定顺序是：tensor/gradient smoke → cache audit/rebuild → 两种 A0 三 fold →
`best_design_condition_single` A0 hard gate → A1。原 `all_edges_mean` A0 始终保留为失败对照，
但不再和新候选混在一起决定 hard gate。新 A0 未全折通过时 runner 返回成功状态并保留两套完整
A0 CSV，但不会创建 matcher checkpoint。
A1 只训练参数匹配的 Q/K linear projector 和一个 dustbin logit；SAM/StreamVGGT/pose solver 都冻结，
没有 pose loss。SAM 与 StreamVGGT descriptor 的原始通道数差异通过无参数的 deterministic channel
pool/pad 统一，因此 trainable Q/K 参数量严格相同。

所有 A1 控制固定同一 current/history/slot/UV/valid support：wrong-ID、shuffle-time 和 channel
permutation 只替换 descriptor 内容，不替换候选 UV 或有效点数。输出同时记录
`parameter_matched_qk`、`control_support_exact` 与 `matcher_frozen_exact`，避免再次把 coverage 或
训练中 pose 适配误认为 SAM token 的作用。

若 A0 出现恶化帧，`v90_stage_a_edges.csv` 会记录每条 current-history edge 的 UV bbox/convex-hull
覆盖率、二维协方差比、design condition、双初始化 residual、cheirality 及 relative pose error；
`v90_stage_a_problem_edges.csv` 只保留恶化帧所属 edge，并由一键命令直接打印，便于复制诊断。

### 12.4 A0 history-edge 诊断与 A0-H（保留原失败结果）

首轮 `local32_same_support_oracle / all_edges_mean` 已运行。short 通过，但 medium 的 frame 450 与
long 的 frame 495 被坏 history edge 拉差，所以全折 gate 为 0。该结果保留，不能覆盖或改写：

| current | history | design condition | relative aggregate | 结论 |
|---:|---:|---:|---:|---|
| 450 | 345 | 416 | 9.29 → 1.02 | 好 edge |
| 450 | 435 | 1722 | 22.29 → 57.42 | 坏 edge |
| 495 | 405 | 32.7 | 16.14 → 0.54 | 好 edge |
| 495 | 465 | 3454 | 22.07 → 103.17 | 坏 edge |
| 495 | 480 | 3067 | 13.46 → 40.51 | 坏 edge |

对应数量、Sampson RMSE 与 cheirality 在这些 edge 上都不能稳定区分好坏；唯一明确且不读取 GT
pose error 的观测量是 normalized eight-point design matrix 的 condition/rank。由此新增一个、且只
新增一个 A0-H 候选：

```text
best_design_condition_single:
  1. 仍生成原来全部严格因果 history edge；
  2. 只在数学求解成功的 edge 中选择 design_condition 最小的一条；
  3. condition 相同时依次选择 design_rank_ratio 更大、history index 更小者；
  4. 没有成功 edge 时保持 inactive/L0 fallback；
  5. 不读取 GT pose error，不设置 residual/count/effect 阈值。
```

CSV 新增 `edge_selection_policy`、`pose_edges_used`、选中 history index/frame 和每条 edge 的
`selected_by_policy`，因此服务器结果可直接确认选择路径。A1 的 SAM、StreamVGGT patch、uniform、
trained-off 与全部扰动分支固定使用同一个 `best_design_condition_single` 算法；只有该 A0-H 在
short/medium/long 全折通过才允许训练 A1。这个修改不增加 token 数、不改变 history 候选、不扫描
阈值，也不声称 metric center 或 absolute trajectory 改善。

### 12.5 A0-H/A1 最终结果（2026-08-06）

`best_design_condition_single` A0-H 已在 short、medium、long 三个严格时间 fold 全部通过：

| fold | active | raw aggregate | refined aggregate | gain | worse frames |
|---|---:|---:|---:|---:|---:|
| short | 4/4 | 13.737° | 1.279° | 90.69% | 0 |
| medium | 4/4 | 13.801° | 1.707° | 87.63% | 0 |
| long | 4/4 | 25.748° | 6.877° | 73.29% | 0 |

这证明：在实例内 current local32 query 已知正确的连续 history UV，且只选择 design condition
最好的因果 history edge 时，固定 O-R1 solver 可以稳定改善 relative rotation 和 translation
direction。A0-H 不读取 SAM descriptor，history 对应也不是从实际离散 32-key 中选择，因此它是
solver/query-support 上界，不是 SAM-token 证据。

A0-H gate 通过后，A1 按预注册协议实际训练了 800 steps。正常 SAM 分支结果如下：

| fold | train loss | held-out PCK@8px | EPE | raw aggregate | refined aggregate | worse frames |
|---|---:|---:|---:|---:|---:|---:|
| short | 5.153 → 0.429 | 2.59% | 142.37 px | 17.886° | 128.781° | 4/4 |
| medium | 4.452 → 0.548 | 0% | 284.45 px | 19.761° | 111.955° | 4/4 |
| long | 4.098 → 0.595 | 0.80% | 321.93 px | 25.988° | 154.722° | 4/4 |

训练 objective 下降 85%–92%，但 held-out matching 与 pose 同时失败。正常 SAM 没有稳定超过
StreamVGGT patch、uniform、trained-off、wrong-ID、shuffle-time 或 channel-permutation 控制；其中
channel permutation 在三 fold 的 EPE 都低于正常 SAM。故当前结果支持以下边界：

- 正确实例内二维 correspondence 能改善 relative pose：已证实；
- 当前 `detector_fpn2 + farthest-UV local32 + Linear Q/K` 能产生这种未来对应：未通过，在当前
  协议下被证伪；
- 仅凭训练 loss 下降判断 SAM token 学成：被证伪；
- metric center、absolute trajectory、跨 scene 泛化：本实验未回答。

后续若继续，只先做 `A0-Q` 离散 history-key oracle 和 train/test matcher audit；不再重复扩大 pose
head、fusion MLP、token 数或 solver sweep。
