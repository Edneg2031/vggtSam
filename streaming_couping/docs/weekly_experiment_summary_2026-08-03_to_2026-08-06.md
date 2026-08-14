# StreamVGGT + SAM3.1：V4–V9.8 证据归档与 V0 位姿理论设计

> 最后更新：2026-08-14
> 用途：这是唯一保留的研究结论与下一阶段设计文档。历史候选代码可以删除，但已经得到的正、负证据不得删除或改写。

## 1. 当前仓库与结论

- 唯一 active baseline 是 **V0**。
- SAM3.1 负责 prompted multi-instance discovery/tracking、mask、persistent ID 和 late birth。
- StreamVGGT geometry辅助mask prompt/competition；纯RGB/native-QK检索负责selected pose，raw pose保留fallback。
- V0不缓存SAM appearance token、不训练pose model；只声明这一条30帧序列上的training-free pose improvement。
- E0/E1 edge-DTF 与 G0 projective ICP 已从 active code 删除；G0-r2 只有实现和 smoke、没有服务器结果，不能作为成功或失败证据。
- 下一步先完成理论和数据协议，不再用新版本号试探性堆代码。

当前主要数据是 ScanNet++ scene `00a231a370` 的 30 帧长 clip：frame 90–525、step 15。未来测试帧固定为：

| diagnostic window | frames |
|---|---|
| short | 360, 375, 390, 405 |
| medium | 420, 435, 450, 465 |
| long | 480, 495, 510, 525 |

这是一个静态室内扫描。晚帧首次进入视野只证明 late discovery，不证明物体发生真实运动；它只能用于
tracking 因果性和 static-scene no-regression，不能证明 dynamic-mask pose improvement。

## 2. 证据等级与不可混淆的命题

| 等级 | 含义 |
|---|---|
| E0 | 只有实现/运行链路，没有正式科学结果 |
| E1 | 能拟合训练数据，只证明容量 |
| E2 | held-out 有改善，但因果变量未完全隔离 |
| E3 | strict future 超过公平控制，正确输入被破坏后收益消失，并通过预锁判据 |

必须区分：

- **SAM system**：proposal、mask、track ID、置信度和 memory 的整体；
- **SAM mask/identity**：实例区域和 persistent ID，不包含 appearance descriptor；
- **SAM token**：`detector_fpn2`、memory-read 或其它 appearance feature；
- **SAM 帮助 pose**：必须优于 raw、同面积错误 mask 和无 SAM 控制，不能由训练 loss 推断。

截至目前没有“SAM token 改善 StreamVGGT pose”的 E3 证据。存在的是整体 pipeline 的旧 E2 结果、
mask/identity 的局部 E2 结果，以及若干 GT 几何/对应的 solver upper bound。

## 3. V4–V9.8 实验账本

| 阶段 | 实际做法 | 已成立结论 | 没有成立的结论 |
|---|---|---|---|
| V4 | pooled instance appearance/geometry → mask-local DPT residual → depth/pointmap/pose | 完整 learned pipeline 在固定参考实例和配置序列上有效，属于 E2 | token、mask、geometry、pose head 未隔离，不能归因 SAM token |
| V5 | V4 前端 + adaptive pointmap + ray-centre solver | 最终 pointmap/pose 有旧 E2 收益 | 不能单独归因 SAM；不能直接恢复为新 baseline |
| V6 | camera/instance/fusion + direct SE(3) head | 第二 clip 相对 raw loss 改善约 7.63%/10.67%/11.43% | direct head 可记忆帧，非 E3 |
| V7/V7.1 | frozen camera L0 + instance residual/control | camera/instance residual 有容量 | future/cross-scene control 未过，所有 causal instance pass=0 |
| V7.2 | 真正 mask-local SAM `detector_fpn2` token，K=8/16/32 | local token 已真实实现，不是缺失功能 | 22 个正式行 causal local pass=0，geometry-only 更稳定 |
| V7.3 | SAM affinity Key + StreamVGGT geometry Value | 所有分支可过拟合长训练序列 | uniform/trained-off 也可拟合；held-out causal SAM pass=0 |
| V7.4 | dynamic birth、永久 slot、causal latest memory、三 future folds | frame 150 late birth 已实际运行；exact L0 fallback 成立 | 三折 SAM causal pass=0，破坏 SAM 输入没有稳定 damage |
| V7.5 | SAM region/ID + ICP/ray solver，无 token/pose head | short/frozen 有局部 region/identity 收益，E2 | frozen/online region/history 全部 all-fold fail |
| V8 matcher-first | correspondence supervision 与 pose 解耦、matcher 冻结 | 排除了 joint pose-head shortcut | matcher/pose 仍无 all-fold causal pass；no-match 可更好 |
| V8 O1–O2.7 | GT pseudo-match + Kabsch/Umeyama，分解 depth/K/calibration | GT geometry、GT depth+GT K 可工作；SAM-instance affine 比 global affine 有上界优势 | predicted depth/K、scale/affine/Sim(3) 均不能稳定修复 pose |
| V9 O-R1/A0-H | 正确 2D correspondence + fixed essential solver | relative rotation/translation-direction 三折 upper bound 成立 | 不是 metric center、absolute trajectory 或 SAM descriptor 证据 |
| V9.1 | actual local32 history key oracle + learned Q/K audit | 离散 key support 已是瓶颈之一 | A0-Q、trained top-1、soft expectation 全部 fail |
| V9.2 | history key 32/64/128/256 + nearest/mutual/unique | 256 keys 可把 coverage/PCK 提到约 0.994/0.917 | 所有策略 pose all-fold fail，不能继续调 detector-FPN matcher |
| V9.3 | fixed edge，hard/soft coordinate、EPE filter、continuous noise | continuous GT positive control pass | soft-K8、oracle filter 和 0.5 px noise都未全折通过；localized support 极脆弱 |
| V9.4 | 多种 deterministic/spatial RANSAC robust solver | continuous control 仍成立 | soft-K8 与 0.5/1 px noise无 solver 全折通过；localized essential route 终止 |
| V9.5 | equal-count instance/full-image/hybrid spatial support | full-image 与 instance/background 在 0.5–1 px noise 下通过 | 失败原因不是 pixel error 单独造成；宽空间支撑是关键 |
| V9.6 | actual 72×72 grid，hard nearest/soft bilinear K4 | full-grid continuous 与 bilinear K4 通过，亚像素 decoder 表达力足够 | hard grid fail；SAM/bbox/shifted equal-count control 不可行，不能声称 SAM mask 独特有效 |
| V9.7 | dense detector-FPN matcher，future causal evaluation | train prefix 能部分拟合 | future PCK@1 基本为 0、EPE 约 98–111 px，pose 全折 fail |
| V9.8 | SAM3.1 genuine memory-read feature vs memory-off | memory feature 链路和 train audit 已完成 | future PCK@1 三折全为 0；train PCK 也仅约 0.095–0.302，causal pass=0 |

### 3.1 可以继续引用的正证据

1. V0/V7.4 已经支持 future object birth、永久 ID、forward-only memory；不要求第一帧出现所有物体。
2. V4/V5/V6 的完整 learned systems 确实出现过 E2 改善，但不能解释成 SAM token 因果收益。
3. GT 3D geometry + Kabsch、GT depth + GT K、正确且宽分布的 2D correspondence 都能修正 pose，说明 raw
   StreamVGGT 存在可修正空间，solver 也不是普遍不可用。
4. V9.5 证明同样 correspondence 数下，full-image/hybrid 的 Hessian 条件远优于 instance-local support。
5. V9.6 证明 72×72 grid 通过 K4 亚像素坐标可以表达连续对应。

### 3.2 不得再重复的失败路线

- pooled/local/memory SAM token → camera token/direct SE(3) head；
- 只凭训练 loss 降到零声称 token 有效；
- `detector_fpn2 + local32/dense + Q/K` correspondence；
- 将 history key 从 32 扩到 256、mutual/unique assignment 或继续 sweep robust essential solver；
- localized-instance-only essential/PnP；
- 固定 predicted depth/K 的 edge-DTF 或 current-only projective ICP；
- TrackHead-BA 当前 checkpoint/preprocess 契约；
- V0 r3 的 direct learned pose correction：medium center 曾下降 26.28%，all-fold fail。

## 4. E0/E1/G0 删除前的最终证据

### E0/E1 edge-DTF

- E0-r2 可微链路通过梯度检查，但 144 个小扰动试验没有一帧通过 pose-basin gate；平均 rotation/
  translation recovery fraction 为 `-1.974/-1.998`。
- E1 的负梯度方向 100% 降 edge loss、正方向 100% 升 loss，证明实现方向正确；但 joint 和 translation
  三折全部失败。
- `sam_object_excluded_edge` rotation-only 约 11/12 帧改善，仍不足以部署完整 SE(3)，且 bed/wardrobe mask
  不是 dynamic ground truth。

结论：edge loss 有方向信息，但固定有偏 depth 的局部目标不等于 GT pose 方向；该路线终止。

### G0 projective point-to-plane

G0-r1 中 full-image 与 SAM-exclusion 的三个 fold mean center 都小幅改善约 `0.004–0.009 native`，证明
projective point-to-plane 含有局部 translation signal；但：

- medium rotation 分别恶化 `0.589°/0.333°`；
- long 仍有 rotation/center 坏帧；
- 所有 branch all-fold pass=0；
- SAM exclusion 未稳定形成优于 full-image 和 shifted-mask 的因果结论。

G0-r2 hybrid photometric + point-to-plane 只完成代码/smoke，未在服务器评测，证据等级为 E0；现已删除。

## 5. 论文能够支持和不能支持的内容

### DualMap（arXiv:2506.01950）

DualMap 的直接启发不是 pose solver，而是：

- hybrid segmentation frontend；
- object-level status check；
- 能随环境变化更新的 persistent object representation；
- global abstract map 与 local concrete map 分工。

它没有证明 instance mask 会直接提升 camera pose。可迁移到本项目的是“双记忆”：一个保存所有对象状态，
另一个只保存可用于 static-camera estimation 的稳定几何；不能把 DualMap 当作 pose-improvement 论文引用。

### SceneVGGT（arXiv:2602.15899）

SceneVGGT 的 pose/SLAM 来自 sliding windows、overlap anchors、camera-pose transform 对齐和 depth grounding；
instance mask 用于 2D semantics lifting、VGGT tracklet、persistent identity、前景/背景地图分离、change detection
和 navigation。它同样不是“mask 优化 pose”的成功证据。

但它支持 object/background map separation、mask erosion、均匀区域采样、persistent tracklet memory 和 3D
visibility/re-ID 的系统设计。

### 4D3R、4DVGGT-D 与 Selfi

- 4D3R 支持 `motion cue → SAM mask → static point selection → PnP/DBA`，关键是可靠 correspondence、
  confidence 和 joint depth/pose optimization，不是 mask 单独求 pose。
- 4DVGGT-D 支持在第二次 VGGT forward 中抑制 dynamic token key、先稳定 pose 再融合 geometry；它报告
  ATE 改善但相对指标不总改善，因此必须同时报告 ATE/RPE rotation/translation。
- Selfi 支持把 VGGT feature 训练成 geometrically localizable feature，再进入 matching + BA；它需要跨场景
  数据和独立 correspondence 指标，不是单 scene SAM adapter。

## 6. 理论问题：SAM mask 单独能否改善 pose

答案是：**不能保证，甚至一般不可辨识。** 一个实例 mask 只给出像素集合和跨帧 identity；相机运动与
物体运动存在多种组合可产生相似 mask。mask 本身不是 point correspondence，也不是 3D measurement。

SAM 能改变 pose 的合理机制只有两类：

1. **选择/重加权机制**：从已有 geometric/photometric factors 中删除违反 static-world assumption 的
   dynamic measurements，并保留 background 与 static objects；
2. **网络内信息路由机制**：在冻结 StreamVGGT 的第二次 forward 中阻止已确认 dynamic patches 写入/读取
   static camera memory。

令可靠几何 residual 为 `r_j(T)`、Jacobian 为 `J_j`，局部 normal matrix 为：

```text
H = Σ_j w_j J_jᵀJ_j + λH_raw
```

动态点带来有方向的系统误差 `ε_dynamic`。正确 mask 将其权重降为零，可以降低
`H⁻¹ Σ Jᵀε_dynamic` 的估计偏差；但把所有实例都删除也会减小 `λ_min(H)`，使位姿更不稳定。

所以 SAM 帮助 pose 的必要条件是：

1. 场景中确有足够 dynamic residual 会偏置 raw pose；
2. dynamic/unknown classification 的 precision 足够高；
3. 删除后 background + static instances 仍提供宽视野、满秩约束；
4. 剩余 factor 的 correspondence/depth 足够可靠；
5. optimizer 位于可收敛 basin 内。

当前 ScanNet++ clip 不满足第 1 个验证条件；E0/G0 也表明当前 frozen predicted-depth residual 尚未满足
第 4/5 条。因此“多检测几个家具然后全部排除”在理论上反而可能让 pose 更差。

## 7. V0 后续方法设计：实例状态分解，而不是实例全排除

### 7.1 系统数据流

```text
RGB_t
 ├─→ StreamVGGT causal raw pass → T⁰_t, D_t/pointmap, K_t, confidence
 └─→ proposal/discovery → SAM3.1 mask + persistent ID + late birth
                           ↓
                   persistent object registry
             {unknown, static, dynamic, lost/reappeared}
                           ↓
       object memory（全部对象） + static pose memory（可信静态内容）
                           ↓
        frozen StreamVGGT mask-conditioned replay / static factor window
                           ↓
                    candidate refined pose
                           ↓
         geometry feedback → SAM prompt/competition/re-ID
```

参数可以保持冻结；关键改变是 measurement routing 和 memory ownership，不是再训练 token→camera adapter。

### 7.2 发现多少物体

V0 当前已把 `instance_prompts` 扩为 `[bed, wardrobe, picture, mat, chair]`，并把永久 registry 扩为
16 个 slots；它仍只会寻找配置的概念。SAM3.1 的 text prompt `object` 不能被描述为真正 class-agnostic
proposal。合理前端是：

1. 用配置的 open-vocabulary detector/proposal vocabulary 周期性发现候选；
2. SAM3.1 对候选生成 mask，并在后续帧传播；
3. 用 causal IoU、prompt/class、3D proximity 和 visibility 去重/re-ID；
4. 长期 registry 容量先设 16，与现有 recovery capacity 一致；slot 永不复用；
5. **每帧最多 5 个 pose-active objects**，而不是全视频最多 5 个对象。

active-5 是计算预算，不是理论常数，必须做 `K=1/3/5` 消融。不能只按 mask 面积选；建议在不读取 GT 的
条件下按下面的分数贪心选取：

```text
score_k = track_confidence
        × temporal_persistence
        × geometry_confidence
        × motion_or_static_relevance
        × incremental_logdet_pose_information
        × (1 - boundary_uncertainty)
```

选择时对与已选实例的空间覆盖重复施加惩罚。背景始终是一个独立且不可被 active-5 挤掉的 support。

### 7.3 object state machine

每个新对象出生时必须为 `unknown`，不能由 noun class 直接判定 static/dynamic：

```text
newborn/unknown
  ├─ 连续 N 次与背景相机运动一致、3D shape/centroid 一致 → static
  ├─ 连续 M 次出现独立且一致的 2D/3D motion residual → dynamic
  └─ 证据冲突/遮挡 → 保持 unknown

static  --连续运动证据--> dynamic
dynamic --长冷却期静止证据--> static candidate（重新建图，不复用旧几何）
```

部署 cue 至少要组合：raw-pose 下的背景补偿后运动、mask interior 的 3D consistency、track persistence、
depth/visibility confidence。mask 边界先 erosion；newborn/unknown 不写 static pose memory。状态切换使用
hysteresis，禁止单帧翻转。

### 7.4 DualMap 启发下的双记忆

- **Object memory**：保存所有 persistent ID、prompt/class、mask history、birth、state、object-local geometry；
  支持移动、消失和重现。
- **Static pose memory**：只保存 background 与已成熟 static objects 的 keyframes/features/geometry；专门供
  camera estimation 使用。

动态或 unknown 对象可以继续被 SAM 追踪，但不能污染 static pose memory。若 static 对象后来被判为 dynamic，
其旧几何冻结为历史版本，当前运动对象开启新 object-local state；不能悄悄用新位置覆盖 static map。

### 7.5 第一候选：mask-conditioned frozen StreamVGGT replay

比重新做固定-depth edge/ICP 更直接的候选，是借鉴 4DVGGT-D 的两阶段信息路由：

1. raw causal forward 得到 `T⁰_t` 和初始 geometry；
2. 只用当前及过去可知信息确认 dynamic/unknown masks；
3. 第二次 forward 保持 StreamVGGT 参数冻结，在 global temporal attention 中禁止 confirmed-dynamic patch
   作为 camera/static-memory 的 key/value；camera/register/background/static tokens 继续参与；
4. 对 unknown 使用软降权，不立刻硬删除；static objects 保留；
5. 输出 candidate pose，未通过 deployable confidence/observability gate 时 exact raw fallback。

为了保持真正 streaming，过去被污染的无限 KV cache不能直接复用。第二次 forward 应只重放一个 bounded
causal keyframe window，或维护单独的 static KV memory。所有 mask 必须下采样到精确 patch footprint，并对
boundary 使用 unknown band，不能简单 resize 后当真值。

这一候选相对旧实验的关键差异：SAM 不预测 pose、不提供 appearance correspondence；它只决定哪些视觉
measurement 能进入 frozen camera memory。它仍没有理论必胜保证，但与“动态像素违反静态场景假设”的动机
直接对应。

### 7.6 第二候选（仅在第一候选有上界时）：instance-aware sliding-window BA

若 masked replay 在动态数据上产生稳定方向但精度不足，再加入局部 window factor：

```text
E = λraw Eraw(T)
  + Ebackground(T, Xbg)
  + Σ_{k∈static} wk Ek(T, Xk)
  + λdepth Edepth-affine
```

- background 与多个 static instances 提供宽空间 support；
- confirmed dynamic instances 从 camera factor 排除，未来可估计独立 object SE(3)；
- unknown 只进入 robust low-weight cue；
- window 联合优化 pose 与有限结构/depth-affine，自身不能重用 G0 的 frozen-depth current-only ICP；
- correspondence 必须先有独立 PCK/EPE/visibility 证据。没有可靠 correspondence 时不写 BA。

这一步可最终转向 Selfi-style VGGT geometric feature，但必须跨场景训练和独立评估；不再训练 SAM token。

## 8. 预锁消融与通过条件

第一轮不改 V0 selected pose，只生成 candidate：

| branch | 目的 |
|---|---|
| `raw` | 原始 StreamVGGT |
| `replay_unmasked` | 控制第二次 forward/window 本身 |
| `mask_all_instances` | 验证“所有语义实例都排除”是否有害 |
| `confirmed_dynamic_mask` | 主候选，只抑制 dynamic，unknown 软降权 |
| `shifted_dynamic_mask` | 同面积/形状错误位置控制 |
| `oracle_dynamic_mask` | 仅 upper bound，不是部署方法 |

另做 `active_objects=1/3/5`，以及 `background_only` vs `background+static_instances`。所有分支使用相同
frame/window、相同计算预算或明确报告额外成本。

通过条件必须在结果前固定：

1. causal prefix check；candidate generation API 不接收 GT；
2. static ScanNet++ 上 no-regression，late birth/ID/registry 仍通过；
3. 每个动态数据集组的 rotation RPE 和 translation/center 指标都优于 raw 与 replay-unmasked；
4. 主候选稳定优于 shifted mask；oracle 只能说明上界；
5. 报告 mean/median、每帧 worse count、active/fallback rate，不能用 composite loss 掩盖某一 pose 分量；
6. 不允许用 GT pose error 选择 candidate/fallback，也不允许只展示改善帧；
7. 只有多场景 all-fold pass 后，才允许 V0 selected pose 从 raw 改为 candidate。

## 9. 数据要求

当前 ScanNet++ clip 继续负责：

- future birth、persistent ID、causal memory；
- static-object 不应被误判 dynamic；
- masked replay 的 no-regression 和 exact fallback。

真正的主验证至少加入多个带 camera GT、真实运动物体的 RGB-D 序列，例如 Bonn RGB-D Dynamic 与
TUM RGB-D `fr3/walking` 系列，并覆盖：低/中/高 dynamic area、相机静止/运动、遮挡和对象重现。

仅在静态 ScanNet++ 上增加到 5 个家具，最多只能测试 object management 和空间覆盖；即使 pose 改善，也不能
归因于 dynamic removal。

## 10. 实现前的锁定结论

1. **不是物体数量越多越好。** 关键是状态正确、空间信息互补和 Hessian 可观测性；错误地排除 5 个静态
   家具可能比只跟踪 2 个更差。
2. **SAM mask 不是 pose measurement。** 它是 measurement ownership/gating；真正 pose information 仍来自
   StreamVGGT 图像/几何及后续可靠 correspondence。
3. **最合理的第一实现不是新 BA。** 先做 frozen StreamVGGT 的 causal dynamic-mask replay，验证 mask
   information routing 是否产生跨动态场景的可归因收益。
4. **DualMap/SceneVGGT 提供系统结构启发，不提供 pose 成功证明。** 本项目必须自己完成 raw/unmasked/
   dynamic/shifted/oracle 的因果验证。
5. 在上述设计进入代码前，V0 保持 raw selected pose；不会恢复已删除的 E0/E1/G0 或 token→camera 路线。

## 11. 参考论文

- [DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes](https://arxiv.org/abs/2506.01950)
- [SceneVGGT: VGGT-based online 3D semantic SLAM for indoor scene understanding and navigation](https://arxiv.org/abs/2602.15899)
- [EPO: Boosting 3D Foundation Models with Edge-based Pose Optimization](https://arxiv.org/abs/2607.00579)
- [4D3R](https://arxiv.org/abs/2511.05229)
- [4DVGGT-D](https://arxiv.org/abs/2605.12027)
- [Selfi: Self Improving Reconstruction Engine via 3D Geometric Feature Alignment](https://arxiv.org/abs/2512.08930)

## 12. 静态场景最小候选：SAM region identity + 3D–2D PnP

用户确认当前目标场景是静态的，因此第 7–10 节的 dynamic-mask replay 不作为当前实现。SAM mask 在静态场景
不能通过“排除动态内容”改善 pose；当前验证的是更窄、可直接归因的命题：persistent region identity 能否删除
通用局部特征中的跨物体误匹配，从而改善冻结 StreamVGGT pointmap 驱动的 PnP pose。

```text
processed RGB history/current
  → SIFT mutual-ratio 2D matches
  → StreamVGGT raw-pose broad reprojection gate
  → SAM eroded region label: background↔background or same-ID↔same-ID
  → historical StreamVGGT world pointmap supplies 3D
  → spatially uniform equal-count support
  → PnP-RANSAC + inlier LM
  → held-out reprojection acceptance and bounded raw fallback
```

长期 registry 仍保留 16 slots 以支持 late birth；每帧按当前 SAM score、mask area、slot 顺序确定性保留最多
5 个 pose-active regions，background 不占这 5 个名额且始终保留。

固定消融为 `full_image_match / sam_region_identity / shuffled_instance_identity`。shuffled control 保持 mask
位置、形状和面积，只轮换 current-frame persistent ID；三组锁定相同 correspondence count。SAM appearance token、
pose training、pose loss 均不使用。候选生成 API 不含 GT 字段，GT 只在全部候选固定后评分。

所有 mask 必须使用缓存中已经过精确 StreamVGGT crop/resize 的 `tracking_masks_stream`，禁止把 SAM 原图输出
直接 resize 后参与匹配。首轮仍只生成独立 candidate；即使单 clip 通过，也只说明该场景的 region-filtering
候选可行，不能宣称跨场景提升，且不能替换 V0 selected raw pose。

运行命令：

历史命令 `commands_v0_sam_region_pose_candidate.txt` 已随失败实现删除；以下结果仅作证据归档。

## 13. 静态场景 SAM region-ID + SIFT/PnP 实验结论

服务器完成 `v0_mesa_style_sam_region_sift_pointmap_pnp_r1`。配置为五个 prompts、16-slot registry、
每帧最多 5 个 pose-active regions；实际发现 4 条 tracks。V0 selected pose 始终保持 raw StreamVGGT。

结果为：

- short fold 的 `sam_region_identity` 仅 1/4 帧生成被接受的候选，center/rotation gain 分别只有
  `0.117% / 0.740%`，fold pass=0；
- 该唯一接受帧是 frame 390，但其 selected instance correspondences=0，结果与
  `shuffled_instance_identity` 完全相同，因此改善来自 background support selection，不能归因于 SAM；
- frame 405 虽有 13 条 instance correspondences，但三组 equal-count 被 shuffled control 锁到 26，低于预锁
  minimum 32，没有生成候选；frame 360 只有 5 条 instance correspondences，且 PnP update 超出部署边界；
- medium/long folds 的 primary active frames 都是 0/4。后半序列每帧 pool 只有 4–42 条对应，大多不足以
  支持固定 PnP/held-out gate；
- 三折 `primary_all_fold_pose_pass=0`，`primary_all_fold_sam_causal_pass=0`。

因此已经证伪的是：**在当前稀疏时间采样、当前五类 SAM regions 和 SIFT mutual-ratio 对应下，使用
persistent region identity 过滤 StreamVGGT pointmap 3D–2D matches，不能形成可部署、可归因于 SAM 的 pose
改善。** 这不证明所有 mask-guided pose 方法均无效，但明确排除了继续调低 PnP minimum、放宽 update bound
或围绕这一 SIFT 后端调参；那些改动既不能补足 medium/long correspondence，也不能让 background-only gain
变成 SAM 因果收益。

本候选保留为独立失败实验，不进入 V0 selected pose。若以后重开 mask-guided pose，前置条件应是一个独立
验证过、在 medium/long folds 仍有足量且准确对应的 generic matcher；先比较 matcher raw 与 region-gated
correspondence precision/coverage，再允许进入 pose。不能直接把 SAM token 接回 camera token，也不能把当前
结果描述成 SAM pose improvement。

## 14. V0 SAM persistent-memory → native StreamVGGT KV 检索协议

该候选采用 RetrieveVGGT 的 training-free context construction，而不恢复已经证伪的 SAM-token camera fusion。
SAM3.1 hidden/appearance feature不进入 StreamVGGT；V0 causal persistent track registry只提供 masks 与 ID。
StreamVGGT 首个 global block 使用经过 QK norm/RoPE 的 current Q 与逐历史帧 native K：global control在全图
patch上池化，SAM branch则分别在 current/history 同一 persistent-ID mask内池化后计算 frame relevance。第一层
选择的帧索引在后续24层复用，选中帧始终贡献完整图像 KV，不裁成实例局部 support。

锁定分支为 `raw_full_history / retrieve_qk / sam_gated_qk / sam_hybrid_qk /
shuffled_instance_memory`。除 raw 外，历史预算均为 first-frame anchor 1 + retrieved 4；hybrid预留2个槽位给
same-ID masked-QK Top帧，缺额由 global QK顺序补齐。所有 candidate先完整生成，GT pose/pointmap只在
之后评分。raw repository replay必须与原 rolling full-history cache通过数值等价检查；候选不会修改 V0 selected
pose。

r1 已完成且 raw replay 三项 max difference 均为 0，但 `retrieve_qk / sam_gated / sam_hybrid / shuffled`
四支输出完全相同。原因不是模型数值错误，而是“当前帧与历史帧是否共享任一实例”在该静态场景中几乎总为真，
该 binary gate没有改变最终有效检索（候选集合退化或 Top选择与 global-QK重合）。因此 r1 证伪了 binary whole-frame
same-instance gate的可识别性，不能据此声称 SAM memory有效或无效。r2 改用上述 same-ID mask-pooled native
QK；shuffled control使用相同 masks/计算量但循环错配历史 ID，使正确/错误 identity必然是可区分干预。

r1 的四个非 raw 分支共同结果为：short 的 center/R/pointmap gain = `15.6335% / 11.8429% /
6.1630%`；medium = `19.0376% / 21.0348% / -34.2773%`；long = `19.0775% / -5.8340% /
-41.5120%`。这给出一个独立但有限的正结果：固定预算 native global-QK retrieval 能稳定降低三折 camera-center
error；同时它没有通过完整 pose/geometry gate，且不能归因于 SAM。medium/long pointmap大幅退化以及 long rotation
退化明确禁止把该分支直接部署为 V0 输出。

SAM memory causal pass要求：primary三折 center、rotation、固定 raw-reference Sim(3) 与固定 raw-confidence
support上的 paired pointmap RMSE均改善，
逐折优于等预算 `retrieve_qk`，并且 shuffled identity逐折破坏这三项收益。运行：

历史命令 `commands_v0_sam_memory_retrieval.txt` 已随失败实现删除；以下结果仅作证据归档。

r2 已完成并构成有效证伪。raw pose/point/confidence replay 的 maximum absolute difference仍全部为 0；selection
intervention audit通过：`sam_gated` 相对 global-QK有23帧不同，hybrid相对 global-QK与 shuffled-ID各有20帧
不同，因此 r1 的“分支未真正分开”问题已经排除。

`sam_hybrid_qk` 的 center/R gain 在 short、medium、long 分别为 `17.4730%/14.3846%`、
`20.4084%/17.0375%`、`7.3359%/1.0037%`，说明重构历史上下文可以改变并在该 clip 上改善 pose。
但对应 pointmap gain 为 `-2.3418%/-53.4971%/-12.8757%`，三折全部退化。更关键的是 shuffled-ID 的
center gain 为 `21.8477%/31.7279%/20.8457%`，三折均高于正确 ID；正确 identity也没有逐折超过
global-QK或 shuffled control。故 pose变化只能归因于固定预算造成的 context truncation/reordering，不能归因于
SAM memory identity，且不能作为 pose+pointmap改进部署。

正确结论分两层：r2 已证伪“正确 SAM identity 是收益来源”的因果命题，因此停止围绕 frame budget、SAM
quota 或 masked-QK ranking做因果调参；但该方法无训练，short/medium/long只是同一序列的诊断窗口，不是
train/test folds。hybrid pose在三个窗口的 center与rotation均改善，因此仍是单序列 engineering pose候选。
最终是否启用改由完整30帧流（frame 90作为gauge reference，其余29帧评分）的整体 pose结果决定。

完整序列评价直接复用已经生成的30帧 causal candidates，不重跑模型：

历史汇总命令 `commands_v0_sam_memory_full_sequence_eval.txt` 已删除；以下结果仅作证据归档。

candidate point-head RMSE不再作为 pose acceptance gate。最终 semantic-map结论必须使用相同 raw depth/K/SAM
masks，分别配 raw pose和candidate pose完成点云融合后再评价；这一步尚未完成。V0 selected pose在完整序列结果
返回前仍保持 raw StreamVGGT。

完整序列汇总已经完成。30帧均经过同一 causal forward，frame 90仅作gauge reference，其余29帧评分；预锁定
`sam_hybrid_qk` 的 center error为 `0.120983 → 0.110489`（`+8.6738%`），rotation为
`2.55064° → 2.36414°`（`+7.3117%`），因此通过单序列 training-free engineering pose gate。
对应 candidate point-head RMSE诊断为 `-9.0035%`，故不得使用candidate pointmap；后续地图融合固定使用 raw
depth/K/masks，只替换pose。shuffled-ID的center更好且r2 causal gate为0，所以成立的表述仅是“固定
SAM-conditioned hybrid pipeline在该序列改善pose”，不能表述为“正确SAM identity导致pose改善”。

鉴于工程目标不要求 SAM identity独立改善pose，正式候选进一步简化为纯 StreamVGGT native-QK retrieval。
candidate generation只读取 `stream_images/frame_indices`，执行 first-frame anchor + first-layer QK Top-4，
运行冻结camera head；不读取SAM mask/ID/memory，不运行point head，也不生成candidate pointmap。完整30帧
干净重放命令：

独立 clean-QK 命令已删除；相同的最小 QK replay 现由 `commands_v0_baseline.txt` 内部执行。

SAM仍负责prompted discovery、persistent tracking、late birth和semantic lifting；若干净QK pose通过，它只替换
语义地图融合使用的camera pose，depth/K继续来自raw StreamVGGT。

干净QK重放已完成：30帧causal stream、frame 90为gauge、其余29帧整体评分。center error由
`0.1209831685`降至`0.1077623591`（改善`10.9278%`），rotation由`2.55063653°`降至
`2.39374542°`（改善`6.1511%`），两项同时通过。candidate generation字段严格为
`stream_images/frame_indices`，SAM pose input、GT candidate field、训练和point head均为0。

因此V0正式selected pose改为`retrieve_qk`；候选artifact缺失时回退raw StreamVGGT。部署仍使用raw
StreamVGGT depth/K/pointmap，禁止使用先前退化的candidate pointmap。允许的结论是“training-free native-QK
history retrieval在这一条30帧序列上改善pose”；SAM identity因果结论和跨场景泛化结论仍为0。下一步只做
语义地图闭环：在完全相同raw depth/K/SAM masks/IDs下分别用raw pose与selected QK pose融合点云。

语义地图A/B现已实现为独立、无训练后处理：地图生成只读取raw depth、raw K、raw/QK pose、SAM
persistent-slot masks/scores与RGB，在StreamVGGT native reference坐标输出两份共享支持点云；GT和固定
raw-reference Sim(3)只在两份地图完成后用于评分。输出保留binary PLY、带prompt/track metadata的PT artifact、
非reference帧paired RMSE及融合点云双向nearest-neighbor/F-score。运行：

```bash
zsh streaming_couping/commands_v0_semantic_map_ab.txt
```

判定只回答“相同raw geometry与SAM语义支持下，QK pose是否改善地图几何”；不把prompt标签当GT语义分类
评价，也不读取已经证实退化的QK candidate pointmap。
