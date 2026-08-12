# SAM3.1 × StreamVGGT：V4–V9.8 证据账本与 baseline 决策

> 更新：2026-08-11
> 数据边界：除特别说明外，全部结果来自 ScanNet++ scene `00a231a370`，只能称为同场景
> 时间外推或同场景第二 clip，不能称为跨场景泛化。

## 1. 最终结论

目前有三类彼此独立的结论：

1. **工程上已经 work**：V7.4 的 SAM3.1 forward-only 动态发现、永久 slot、因果 memory 和
   中途实例 birth 已经跑通。物体不需要全部出现在第一帧；本序列中第三条 track 在 frame 150
   才出生，并能在之后成熟并加入流程。
2. **pose 上界已经 work**：正确且空间分布良好的 2D correspondence，以及 GT camera geometry
   或 GT depth + GT K，能通过固定显式 solver 改善 StreamVGGT relative pose。
3. **真实 SAM feature 尚未 work**：从 pooled token、`detector_fpn2` local/dense descriptor 到
   SAM3.1 temporal-memory feature，均未在严格未来 fold 中建立超越公平控制的 correspondence
   和 pose 因果收益。

因此，当前不存在一个同时满足以下三点的已验证版本：

- 支持未来新物体加入；
- 使用真实 SAM token，而不是 GT correspondence；
- SAM token 在所有未来 fold 因果地改善 pose。

可清理保留的工程 baseline 应从 **V7.4 动态实例前端**出发；其 pose 分支只能称为经验 baseline，
不能称为“已证明 SAM token 有效”。

## 2. 理论判断

### 2.1 SAM 信息何时能影响 pose

相机位姿是 6-DoF 几何状态。若当前/历史图像点为 `u_t`、`u_h`，相对位姿为 `(R,t)`，
真正约束 pose 的是类似下面的空间测量：

```text
2D–2D:  u_h^T K^-T [t]x R K^-1 u_t = 0
3D–3D:  X_h ≈ R X_t + t
2D–3D:  u_t ≈ project(K, T_t, X_world)
```

SAM 的 semantic token、mask 或 persistent ID 本身不包含新的 6-DoF 方程。它最多用于：

- 判断两个观测是否可能属于同一对象；
- 给 correspondence 或 residual 分配权重；
- 选择静态区域、排除动态区域；
- 决定从哪条历史 observation 读取几何。

只有当它最终产生**正确的带坐标对应**，并且对应的空间支撑使设计矩阵满秩且条件良好时，
它才可能改善 pose。把 pooled semantic token 直接送入 SE(3) head，只证明网络能拟合 pose，
不构成可辨识的几何机制。

### 2.2 冻结两个 backbone 不是理论障碍

冻结 SAM3.1 和 StreamVGGT 不会自动导致“没有修正能力”。如果冻结输出能提供准确 correspondence、
K 和 point geometry，显式 solver 不需要训练 backbone 也能修正 pose。V8 的 GT geometry/Kabsch
和 V9 的 GT correspondence/epipolar 正对照已经证明这一点。

当前失败来自冻结输出的**信息质量和可观测性**，而不是“冻结”本身：

- StreamVGGT predicted depth 的局部 shape 和跨帧一致性不足；
- predicted K 也会独立损害三维求解；
- SAM descriptor 没有预测出未来像素对应；
- instance-only correspondence 的空间范围太局部，极几何条件退化；
- 单场景 matcher 的 loss 下降没有转化为 PCK@1。

### 2.3 一个理论上可成立的融合必须满足

1. **因果性**：只读当前及更早 observation；新物体 birth 不能改写过去。
2. **成熟性**：birth 帧只有一次观测，不能形成跨帧约束；至少下一次可靠观测后才可参与 pose。
3. **身份正确**：wrong-ID/time-shuffle 必须破坏结果，否则 persistent identity 没有提供因果信息。
4. **空间可观测**：点必须覆盖足够大的二维范围；不能只集中在一个小实例区域。
5. **坐标精度**：匹配误差必须低于 solver 的实际容差，而不只是通过宽松 PCK@8。
6. **几何一致**：K、depth、坐标系和历史 pose 必须一致。
7. **无直接捷径**：semantic feature 不能经高容量 head 直接记忆 frame→pose。
8. **公平控制**：必须超过 equal-support geometry/full-image、SAM-off、wrong-ID、shuffle-time 和
   parameter-matched trained-off。

V4–V9.8 没有一个真实 SAM-token 分支同时满足这些条件。

## 3. DualMap 能借鉴什么

参考论文：Jiang et al., *DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language
Navigation in Dynamic Changing Scenes*, RA-L 2025, arXiv:2506.01950。

DualMap 的实际数据流是：

```text
posed RGB-D frame
  → object observation {3D points, semantic feature, class, timestamp}
  → local/concrete object map
  → stable anchor + volatile-object relation
  → global/abstract map
```

它把两个职责分开：

- **concrete map** 保存最近、精细、可更新的 object observations；
- **abstract map** 只保存稳定 anchor、语义关系和全局结构，用于候选选择；
- 新 observation 用 `semantic cosine + 3D overlap` 关联，低相似度时创建新 object；
- stability check 和 split detection 负责清理噪声、处理错误合并；
- 局部观察更新全局 abstract map，使移动或新出现的物体可以被重新纳入。

但 DualMap **不估计 camera pose**。论文输入是 posed RGB-D，真实系统使用 FastLIO2 外部定位，
作者也明确把“不做 camera pose estimation”列为限制。因此它不能证明“语义 map 能修正 pose”。

对本项目真正有用的启发是状态解耦：

| DualMap 思路 | 本项目对应设计 |
|---|---|
| global abstract map | persistent object registry：ID、prompt、birth、last-seen、stable/movable、质量 |
| local concrete map | 最近 observation：mask、UV、StreamVGGT geometry、descriptor、timestamp |
| semantic + geometry association | semantic 只辅助关联；pose residual 仍来自显式空间测量 |
| stability/split check | 防止一次误检写入永久 memory，并允许错误 track 拆分/失效 |
| online insert/update | 新物体未来出现时分配永久 slot，不要求第一帧可见 |

最重要的结构原则是：**abstract semantic map 负责“选谁和选哪段历史”，local concrete map 才负责
生成 pose factor；不要让 abstract token 直接回归 pose。**

## 4. 统一实验设置与证据等级

主要长序列是 frame 90–525、step 15、共 30 帧。frame 90 只定义 camera gauge，不要求物体
必须存在。V7.4 之后主要使用三组严格时间 fold：

| fold | 训练前缀 | 未来测试 |
|---|---|---|
| short | 270–345 | 360、375、390、405 |
| medium | 270–405 | 420、435、450、465 |
| long | 270–465 | 480、495、510、525 |

证据等级：

- `E1 capacity`：训练集能过拟合，只证明模块有容量；
- `E2 empirical`：held-out/第二 clip 有改善，但未隔离 SAM token；
- `E3 causal`：未来 fold 超过公平控制，且输入破坏稳定毁掉收益。

截至 V9.8，没有 SAM appearance/local/memory feature 达到 E3。

## 5. V4–V9.8 实验账本

| 版本 | SAM 如何参与 | 实际结果 | 结论 |
|---|---|---|---|
| V4 | pooled appearance + mask/ID + geometry 形成 instance token；修改 camera/DPT patch，再用 ray solver | 90–240 ATE `0.36879→0.20464`；492–589 `0.36079→0.33316` | 完整系统 E2；固定参考实例，未隔离 token |
| V5 | V4 learned pointmap + bounded SO(3) + adaptive support/ray-centre | 90–240 ATE `0.36879→0.15932`；第二 clip `0.36079→0.35361`，低 support 时 pointmap exact raw | 当前旧完整系统中效果最稳，但使用 reference-frame GT instance 初始化，不支持动态 discovery 结论 |
| V6 | pooled appearance/geometry/camera → direct SE(3) head | held-out 和第二 clip 有部分改善；所有分支训练可到近零 | E1/E2；直接 head 与混合输入不能证明 SAM token |
| V7 | pooled/cross-attention/identity-Key/geometry-Value/local geometry 结构梯度 | identity/local 分支未稳定超过 camera L0 | 未通过因果验证 |
| V7.1 | 冻结 L0 后训练 global instance residual | 所有 `causal_instance_pass=0` | development 改善可由 camera capacity 解释 |
| V7.2 | 真正 mask-local `detector_fpn2`，K=8/16/32 | 22 个正式行全部 `causal_local_pass=0`；geometry-only 更稳定 | local token 未证明有用 |
| V7.3 | SAM Q/K affinity 搬运 StreamVGGT geometry Value | 全长可过拟合；held-out `causal_sam_pass=0` | SAM 可当训练帧编码，不等于物理 correspondence |
| V7.4 | V7.3 fusion + forward-only dynamic birth + three future folds | SAM+geometry 相对 L0 为 `+22.81/+10.26/+22.01%`，但 short/medium 不如 geometry control，且扰乱 SAM 常不变差；all-fold pass=0 | 动态系统 work；SAM-token 因果失败 |
| V7.5 | SAM mask/ID 选点；ICP/ray，无 token、无 pose head | 只在 short/frozen 的 region/history 通过；all-fold 均失败 | region/ID 有局部 E2，非稳定 pose 方法 |
| V8 matcher-first | GT-world match supervision；冻结 matcher；evidence-only pose | short 局部改善，medium/long 无 all-fold pass；no-match control 可更好 | 更严格结构仍未证明 SAM token |
| V8 O1–O2.7 | GT pseudo-match + Kabsch/geometry factorization，无训练 | GT camera+GT history、GT depth+GT K 通过；predicted depth/K、scale/affine/Sim3 均失败 | 显式三维 solver 正确；预测几何是瓶颈 |
| V9 O-R1/A0-H | GT visible 2D correspondence + fixed epipolar solver | continuous GT 三折通过，rotation/translation-direction 明显改善 | 正确 2D correspondence 的 pose 上界成立，非 SAM 证据 |
| V9.1–V9.2 | 实际 local32/history 32–256 key、nearest/mutual/unique | 256 key coverage `99.45%`、PCK@8 `91.68%`，pose 仍失败 | token 稀疏和重复 collision 不是最终瓶颈 |
| V9.3–V9.4 | 固定 edge；soft K8/noise；多种 robust solver | instance-local 0.5px 已失败，所有 robust solver 无 all-fold pass | 终止 localized-instance essential 路线 |
| V9.5 | equal-count full-image 与 instance/background hybrid | 两种 expanded support 在 0.5/1px 均三折通过、零恶化 | 真正瓶颈是空间条件；实例证据不能独占 support |
| V9.6 | 实际 72×72 grid；hard 与 soft-bilinear K4 | full-grid hard 失败；soft-K4 零 EPE 且三折通过 | 坐标表示可行，但需要 sub-grid decoder |
| V9.7 | dense `detector_fpn2` matcher，只训 2D label | future PCK@1 `0/0/1.65%`，pose 全折负收益 | dense detector feature 在当前协议下失败 |
| V9.8 | 真实 history-read SAM3.1 temporal-memory map | future PCK@1 三折全 0，EPE `120.46/117.12/117.47px`，全部差于 raw control | temporal memory 在当前单场景 matcher 下失败 |

### V9.8 train/future 审计

| fold | feature | train PCK@1 | train EPE | future PCK@1 | future EPE |
|---|---|---:|---:|---:|---:|
| short | sam memory | 30.21% | 1.47px | 0 | 120.46px |
| medium | sam memory | 11.41% | 2.89px | 0 | 117.12px |
| long | sam memory | 9.49% | 9.52px | 0 | 117.47px |
| short | same-call memory-off raw | 25.26% | 1.55px | 0 | 116.85px |
| medium | same-call memory-off raw | 12.66% | 2.08px | 0 | 99.63px |
| long | same-call memory-off raw | 10.38% | 7.37px | 0 | 99.35px |

loss 下降 85%–95% 仍没有形成训练内高精度 correspondence；因此 V9.8 不只是“训练学会后时间
过拟合”，而是 objective/decoder 在训练内就未可靠学到 PCK@1，未来又完全崩溃。

## 6. 已经证实

1. V4/V5 的**完整 pipeline**在固定参考实例、同场景 clip 上确有 E2 效果。
2. V7.4 的多 prompt、多个实例、永久 slot、forward-only dynamic birth 和无未来回看已实现。
3. 新实例 birth 帧只建立 observation/memory；下一次可靠观测后才能形成 pose evidence。
4. 无 mature instance 时 exact L0 fallback 可以实现；这属于可观测性条件，不是读取 GT 指标挑帧。
5. GT 3D geometry + Kabsch 与 GT depth + GT K 可以修正 pose。
6. 正确且全局/混合空间分布的 2D correspondence 可以显著修正 relative rotation 和
   translation direction，并能容忍 1px 噪声。
7. 实际 72×72 grid 可以用 K4 soft coordinate 表达连续 history UV。
8. SAM mask/ID 含有一定 region/identity 信息，但只有局部 fold 证据。

## 7. 已经证伪或必须停止重复

1. adapter 把 pose loss 拟合到零不能证明 SAM token 有用；uniform、geometry、SAM-off 也可过拟合。
2. pooled/global token 直接回归 SE(3) 没有排除 frame memorization。
3. `detector_fpn2 local token → affinity → geometry Value → pose head` 在 V7.2–V8 已重复失败。
4. 增大 history key 到 256、mutual/unique assignment 不能修复 pose。
5. localized-instance-only essential solver 对亚像素误差过于脆弱；V9.4 已满足终止条件。
6. predicted depth/K 的三维路线不是简单 scale、offset、affine 或 Sim3 问题。
7. dense detector-FPN 和真正 temporal-memory feature 都未学出未来 correspondence。
8. 当前结果不能支持 metric center、absolute trajectory 或跨场景 SAM 因果结论。
9. 清理后的 V0 r1/r2 都是 no-op：真实 cache 上 camera train loss
   `0.344872→0.344872`，r2 的 1200 step 全部梯度严格为 0；但随机输入 smoke 能正常下降。
   这把问题定位为真实 `camera_hidden` 下的零初始化 stationary point。r1/r2 已废弃，不能作为
   baseline 证据。

## 8. V0 r3 pose 候选（已废弃）

历史实现修订为 `v71_pose_conditioned_l0_v74_geometry_r3`：

```text
RGB stream
├─ frozen StreamVGGT → raw pose + pointmap/local geometry
└─ SAM3.1 forward-only prompt sessions（不提取 appearance token）
     → dynamically discovered persistent IDs
     → permanent slot registry
     → birth 后用历史 object geometry 生成 box/positive prompts
     → raw/corrected mask 固定 compete gate
     → corrected mask/ID/quality observation
          ↓
recent concrete observation memory
     ├─ V7.1-style camera hidden + raw relative L0 pose → V0 selected pose
     └─ V7.4-style geometry-only transport → 单独计分的 candidate
          （无 mature/static/geometry-valid slot 时 exact L0）
```

实现边界：

- SAM3.1 的职责仅是 open-vocabulary discovery、mask、persistent ID 和 slot lifecycle；缓存明确
  `cache_sam_appearance: false`，pose 网络不接收 SAM appearance/local/memory token。
- 多 prompt 中每个 prompt 可发现多个实例；slot 在首次出现时永久分配，birth 帧只写 memory，下一次
  可靠观测后才可能进入 pose。
- 只有 `identity_valid + track_valid + geometry_valid + static_valid` 的实例可写入/读取 pose memory；
  moving/unknown/低质量实例被排除。
- 几何纠 mask 使用 frozen StreamVGGT raw pose/pointmap，不读 refined 同帧 pose，避免同帧循环依赖；
  correction 进入下游 observation，但暂不回写 SAM3.1 multiplex 内部 memory。
- V0 的最终 pose 选择 V7.1-style L0。为避开 r2 在真实 cache 上的零梯度 stationary point，L0
  同时读取 raw StreamVGGT 相对参考帧 pose；该输入部署时可得，不读 GT、SAM token 或未来帧。
  geometry transport 不再静默覆盖最终结果，只作为 `geometry_candidate` 输出。
- runner 验证 prefix causality、late birth、birth-frame inactivity、moving-object exclusion 和 inactive
  exact fallback；默认用 105–255 训练 camera，270–345 训练 geometry candidate，360–525 只评估。
- runner 拒绝零梯度、零参数更新和训练 loss 不下降；评估主指标固定为 mean camera-center error，
  pose loss 只作为 secondary。12 个未来帧固定分成 short/medium/long 三折，并同时报告每折均值和
  worse-frame 数量。只有每折 center 均改善且没有任何 center 坏帧才可验收。
- 同参数量控制固定为 `camera hidden + raw pose`、`raw-pose-only`、`camera-token-only` 和
  `time-only`。这用来判断收益是否真的来自 StreamVGGT camera hidden 的增量，而不是 raw pose
  轨迹输入或单场景 frame/time 拟合。

### V0 r3 逐帧复审结论

服务器旧 summary 的 `baseline_acceptance_pass=1` 只读 12 帧 aggregate pose loss，不能作为正式
验收。按同一份 `frame_diagnostics.csv` 重新计算 center 主指标：

| fold | frames | raw center | selected center | center gain | worse frames | robust pass |
|---|---|---:|---:|---:|---:|---:|
| short | 360–405 | 0.130182 | 0.0904655 | 30.5087% | 1 | 0 |
| medium | 420–465 | 0.0973828 | 0.122970 | -26.2753% | 3 | 0 |
| long | 480–525 | 0.234801 | 0.128532 | 45.2593% | 0 | 1 |

因此当前 V0 r3 的正式结论是 **all-fold fail**。12 帧 aggregate center 的确改善，但它被 short/long
折主导，掩盖了 medium fold 的明显下降；这只能记作同场景 aggregate E2 diagnostic，不能称为稳定
pose baseline。

geometry candidate 也不能升级为 selected：它相对 selected 在 short 改善约 11.50%，medium 没有
active correction，long 下降约 1.49%；frame 525 的 center 和 rotation 都进一步变差。当前证据不支持
继续调 geometry transport 阈值。

### V0 r3 参数匹配控制最终结论

四个分支均为 `996,126` 参数，使用相同训练帧、步数、seed、loss 和未来 fold：

| input | short center gain | medium center gain | long center gain | all-fold pass |
|---|---:|---:|---:|---:|
| camera token + raw pose | 30.5087% | -26.2753% | 45.2593% | 0 |
| raw-pose-only | 24.1559% | -9.24560% | 32.2474% | 0 |
| camera-token-only | 29.3107% | -7.45889% | 47.0595% | 0 |
| time-only | -3.63138% | -33.3661% | 20.2059% | 0 |

训练审计不是 no-op：normal、pose-only、camera-token-only 的训练 loss 都下降约 100%，time-only 下降
38.83%。然而所有方法都在 medium 的 4 帧中产生 3 个 center 坏帧。最终 decision 为：

- `normal_all_fold_pass=0`；
- `normal_beats_pose_only_all_folds=0`；
- `normal_beats_camera_token_only_all_folds=0`；
- `normal_beats_time_only_all_folds=1`；
- `v0_pose_validation_pass=0`。

这证实 camera hidden 和 raw pose 都含有可拟合信息，normal 也不只是 time index 拟合；但它没有建立
稳定的未来 pose 修正，也没有证明两种输入的融合优于单输入。camera-token-only 在 medium/long 的
center 反而优于 normal，说明加入 raw pose 会在部分时间段造成负迁移。**到此终止 direct learned
SE(3) head 路线，不再调 hidden dim、训练步数、loss 权重或 geometry transport 阈值。**

## 9. 当前保留的 V0 r4

当前 active 实现修订为 `raw_streamvggt_dynamic_tracking_r4`：

```text
RGB stream
├─ frozen StreamVGGT → raw pose + pointmap/local geometry
└─ SAM3.1 forward-only prompt sessions
     → multi-prompt / multiple instances
     → persistent ID + permanent slot registry
     → causal late birth + mature observation
     → historical geometry 生成下一时刻 prompt
     → raw/corrected mask competition

selected pose = raw StreamVGGT exactly
```

清理结果：

- 删除 `CameraPoseBaseline`、`DynamicInstanceGeometryRefiner` 及训练 runtime；
- 删除已完成使命的 parameter-matched control runner；其结果保留在本账本；
- 不再生成或读取 `camera_baseline.pt`、`geometry_pose_refiner.pt`；
- 每次 r4 audit 会删除这两个旧 checkpoint 以及旧 `validation/v0_pose_control_*` 文件，防止把 r3
  控制结果误认为 r4 输出；
- `poses.pt` 只保存 raw 和与其完全相同的 selected pose；
- summary 固定记录 `pose_modification_applied=false`、`pose_improvement_claim=false`；
- tracking gate 验证 birth registry 与逐帧 discovered 数严格一致、discovery 单调、mature track 只能读取
  更早的 geometry birth、`(prompt, track ID)` 唯一，并要求至少一个未来 birth。

因此 V0 r4 的 `tracking_baseline_acceptance_pass=1` 只表示流式动态实例工程不变量通过，不表示 SAM
tracking 精度或 pose 精度提高。几何 correction 的应用次数会被记录，但没有 GT mask 对照时也不能把
“应用了 correction”写成“tracking accuracy 已提高”。

运行入口：

```text
zsh streaming_couping/commands_v0_baseline.txt
```

输出目录固定为 `outputs/streaming_couping_v0/`，并复用已有 V7.4 dynamic cache。结果只包含
`baseline_summary.json`、`frame_diagnostics.csv`、`dynamic_instance_diagnostics.csv` 和 `poses.pt`；旧
pose checkpoint 即使仍在服务器输出目录也不会被 active code 读取。

这个 baseline 符合“第一帧不需要出现所有物体，未来物体可加入”，但仍有三个限制：

1. SAM3.1 是 open-vocabulary prompt detector，不是 class-agnostic all-object proposal；目前只配置
   `bed`、`wardrobe`，会发现这两个概念的多个实例，但不会自动覆盖任意类别。
2. selected pose 是 raw StreamVGGT；当前没有 pose 改善方法。
3. 当前 runner 是按时间顺序的 causal replay；SAM3.1 session 仍由 cache builder 对整段有序帧创建和
   关闭，尚未封装成常驻服务式的 `step(frame)` API。

## 10. DualMap 启发下值得保留的下一条理论路线

V0 三折失败后，不应继续扩大或调参当前 direct learned SE(3) head。下一版代码应保留 V0 的动态
实例前端和 geometry-assisted mask，但把 raw StreamVGGT pose 恢复为未验收阶段的 selected 输出，
把新 pose 方法始终作为单独 candidate。候选路线是：

```text
global abstract registry
  = object identity / prompt / birth / stability / movable status

local concrete factor window
  = full-image or background correspondences
  + equal-count static-instance correspondences
  + UV / geometry / timestamp / uncertainty

pose update
  = calibrated-K epipolar factors
  + StreamVGGT relative-pose/translation-scale prior
  + explicit causal sliding-window optimizer
```

SAM 的可验证贡献改为：

- 用 mask 排除移动物体；
- 用 persistent ID 选择长期静态 anchor 的 history；
- 用 stability/split 状态阻止错误 observation 污染 memory；
- 在全图支撑中做 instance/background 分层，而不是只在实例内部求 essential matrix。

实现顺序固定为：

1. 定义 `CausalCorrespondenceBackend`，接入真正为视频匹配/跟踪预训练的冻结 correspondence source；
   不再重复 V9.7/V9.8 已失败的 SAM detector/memory descriptor matcher。
2. 同一批 correspondence 上运行 full-image、SAM dynamic-excluded、SAM instance/background-stratified、
   random/bbox mask 四个 equal-count 分支。
3. solver 只优化宽基线可观测的 rotation/translation direction；metric translation scale 由 frozen
   StreamVGGT relative pose prior 锚定，避免重新进入 V8 已失败的 predicted-depth Kabsch 路线。
4. 固定权重后先过当前长序列三折，再跑同场景第二 clip；在此之前不覆盖 raw pose，也不设计基于
   GT 指标的回退阈值。

该路线的必要控制是 equal-count full-image、random/bbox mask、wrong-ID、shuffle-time、SAM-off。
只有 SAM-stratified 分支在相同 correspondence 和 solver 下稳定超过这些控制，才能声明
“SAM system 帮助 pose”。这首先验证 mask/identity/status，而不是强行证明 appearance token。

## 11. V0 r5 候选实验：因果 TrackHead + 固定结构 BA

V0 r4 的职责判断保持不变：SAM3.1 是主要 tracking 前端；StreamVGGT 几何在 tracking 分支中主要用于
低质量 mask 的候选纠正/回退。但 raw StreamVGGT pose 存在系统偏移，因此新增一个与 tracking 解耦的
pose candidate，专门验证冻结的时序点轨迹能否修正这个偏移：

```text
V0 r4 cache（不重跑 backbone）
├─ SAM3.1 mask / persistent ID / birth / static-quality
│    └─ 排除身份未知区域，或做静态实例/背景等量分层
└─ StreamVGGT 四层 causal DPT token + image
     └─ frozen built-in TrackHead：最近连续历史，最多 5 帧、256 个全图点轨迹
          └─ anchor raw depth + raw K/pose 反投影为冻结 3D 点
               └─ bounded motion-only fixed-history BA（只优化当前相机）
                    ├─ raw StreamVGGT pose 初始化
                    ├─ raw pose/temporal prior
                    └─ 只输出 candidate，不改写 r4 selected pose
```

这个实验与已有失败路线的区别是：

- 不再用 pooled SAM token 或高容量 SE(3) head 直接回归 pose；
- 不训练 matcher 或 pose model，correspondence 来自 StreamVGGT 自带的视频 TrackHead；
- 不做已经被 V9.3–V9.4 证伪的 instance-only essential matrix，而使用 V9.5 已通过上界的宽空间支撑；
- 不做 V8 已失败的跨帧 predicted pointmap 3D–3D Kabsch；raw depth 只在 anchor 帧生成固定 3D landmark；
- 每个 evaluation frame 只输入最近连续历史、合计最多 5 帧的 causal token；历史相机固定，
  只优化当前相机，禁止一次读取完整未来序列；
- 新物体仍可未来 birth，但 birth 本身不会改过去 pose；只有形成可信历史 observation 后，其 mask 状态
  才能参与后续选点。

预先锁定的主方法是 `sam_dynamic_excluded`：保留全图宽支撑，只排除 SAM 已观测但不能通过静态/身份
可信 gate 的区域。以下同数目控制同时运行，但不会按 GT 结果选择赢家：

| 方法 | 作用 |
|---|---|
| `full_image` | 不读 SAM 的 TrackHead + BA 基线 |
| `sam_dynamic_excluded` | 预先锁定主方法；排除低可信/可能动态区 |
| `sam_instance_background_stratified` | 静态实例和背景各一半 |
| `bbox_instance_background_stratified` | bbox 形状控制 |
| `random_instance_background_stratified` | 确定性随机平移 mask 控制 |

GT 只在全部 candidate 写完后评分。主指标仍是三折 mean camera-center error；每折必须为正收益且
`center_worse_frames=0` 才通过。即使通过，本次脚本也只写
`eligible_for_future_v0_revision=1`，不会在同一次运行中用 GT gate 覆盖 raw pose。若主方法三折通过，
再单独修改 V0 revision 并用已锁配置复跑；若 full-image 通过而 SAM 主方法不通过，只能称 TrackHead/BA
有效，不能称 SAM 帮助 pose；若全部失败，则停止该 candidate，不调 GT 回退阈值。

运行入口：

```text
zsh streaming_couping/commands_v0_track_ba_candidate.txt
```

首次服务器运行五个方法均为 `active_frames=0/12`，candidate 与 raw 完全一致；但这**不能证伪
TrackHead + BA**，因为 BA 从未启动。逐帧原因全部是
`fewer_than_min_correspondences`：anchor 有 `106–227/256` 个有效点，而 current 严格为 `0/256`。
因此当前只证实失败发生在 TrackHead 当前帧 validity gate，且与 SAM mask 无关（`full_image` 同样为
零）；在拆清坐标、visibility 和 confidence 前，不降低阈值、不调 BA 权重。

实现修订 `causal_track_head_fixed_structure_ba_r2_validity_audit` 已在同一运行入口加入轻量诊断。已有
候选输出时，它只重跑 `full_image` 和预锁定的 `sam_dynamic_excluded` TrackHead 窗口，不运行 BA、
不解码 GT，也不覆盖既有 candidate。它逐帧及汇总打印：finite、in-bounds、visibility pass、
confidence pass、gate 交集、geometry 交集和坐标/分数范围，并自动区分坐标尺度/re-anchor、visibility、
confidence 或 gate 交集失败。

服务器仍只需运行：

```text
zsh streaming_couping/commands_v0_track_ba_candidate.txt
```

需要回传该命令末尾打印的 `V0 Track-BA current-validity factor audit`；无需设置
`V0_TRACK_BA_FORCE_RERUN=1`，也无需手写额外命令。
