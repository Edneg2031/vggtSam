# SAM3.1 × StreamVGGT 实验验证账本

> 最后更新：2026-08-05  
> 用途：这是后续方案设计前必须检查的单一历史账本。它记录“已经实现什么、实际跑过什么、
> 结果是否通过、最多能支持什么结论”，避免把旧实验换一个版本号后重复执行。

## 1. 使用规则

设计新实验前必须依次回答：

1. 新问题是否已经在下文“验证项索引”中出现；
2. 新实验相对已有实验究竟改变了哪个因果变量；
3. 是否仍存在可直接记忆帧到 pose 的 residual head；
4. 主结论是否同时包含 parameter-matched control、held-out 时间帧和输入破坏测试；
5. 如果只是在训练帧把 loss 拟合到零，必须标为 capacity，不得标为 SAM 因果证据；
6. 如果所有 clip 都来自 `00a231a370`，必须称为同场景时间外推或第二 clip，不能称为跨场景泛化。

没有填清以上六项时，不增加新的训练版本。

## 2. 证据等级

本账本统一使用四级标签：

| 等级 | 含义 |
|---|---|
| `E0 实现` | 代码或控制项存在，但没有正式结果，或结果尚未回填 |
| `E1 容量` | 能过拟合训练帧，证明模块有表达能力，不证明输入提供了正确物理信息 |
| `E2 相关` | held-out 指标有改善，但没有排除容量、mask、gate、geometry 或直接 pose-head 捷径 |
| `E3 因果` | held-out 超过公平控制，正确输入优于 SAM-off/wrong-ID/shuffle，并通过预先锁定的判据 |

截至 2026-08-05，尚无实验达到“**SAM3.1 local descriptor/token 对 StreamVGGT pose 的 E3 因果证明**”。

必须同时区分以下三个命题：

- `SAM system`：mask、track ID、置信度和 local descriptor 的整体；
- `SAM region/identity`：mask 区域和 persistent ID，不包含 local descriptor；
- `SAM token`：`detector_fpn2` 的局部或池化 appearance descriptor。

geometry-only 分支通常仍使用 SAM mask 和 slot 来规定实例区域，因此它只移除了 SAM descriptor，
不是完全移除 SAM system。

## 3. 总览

| 版本 | SAM 表示/作用 | 已完成的关键验证 | 实际结论 | 等级 |
|---|---|---|---|---|
| V4 | pooled appearance/geometry instance token → mask-local DPT patch residual → depth/pointmap | train/held-out/第二 clip、depth/pointmap/pose 联合监督、module-off | 已经验证过 SAM/实例信息参与 learned depth-shape/pointmap 修正；token、mask、geometry、pose 与 ray solver 未隔离 | E2（整体系统） |
| V5 | V4 learned DPT depth/pointmap 前端 + adaptive pointmap + ray-centre solver | held-out/第二 clip、raw/fixed/adaptive、module-off | 最终 pose/pointmap 有提升；无法归因给 SAM token | E2（整体系统） |
| V6 | pooled SAM appearance + persistent geometry | overfit、held-out、validation、第二 clip、appearance/geometry/shuffle 控制 | instance/fusion 有部分 held-out 改善，但直接 SE(3) head 和混合输入未排除捷径 | E2 |
| V7 | pooled token/单 token attention/identity-Key/geometry-Value/local geometry | temporal、validation、第二 clip、long30、输入扰动 | 重型 SAM/geometry 结构未稳定超过 camera L0 | E1/E2 |
| V7.1 | frozen L0 + global instance residual | camera capacity、gate、appearance、geometry、future/cross | 所有 `causal_instance_pass=0` | 未通过 E3 |
| V7.2 | 真正的 SAM3.1 mask-local token，K=8/16/32 | SAM pool/match、geometry-only、dual、wrong-ID、shuffle、future/cross | 所有 `causal_local_pass=0`；geometry-only 更稳定 | 未通过 E3 |
| V7.3 | SAM affinity Key，StreamVGGT geometry Value | uniform/geometry/SAM/combined、SAM-off、wrong-ID、shuffle、长序列 capacity | 全长可过拟合；held-out `causal_sam_pass=0` | E1；未通过 E3 |
| V7.4 | 动态 birth + causal latest memory | 三个时间 fold、trained-SAM-off、公平 support、输入破坏 | 三个 fold 全失败；正确 SAM token 的破坏通常没有稳定伤害 | 未通过 E3 |
| V7.5 | SAM mask/ID + 显式 ICP/ray solver，无 token、无 pose head | full/bbox/random/stale/wrong-ID、frozen/online history | 局部 fold 有收益，但 region/history 无 all-fold pass | E2（region/ID） |
| V8 matcher-first | 显式 GT correspondence 监督、冻结 matcher、evidence-only pose | geometry/SAM/combined、trained-off、no-match、dual Value、三个 fold | matcher/pose 均无 all-fold SAM causal pass | 未通过 E3 |
| V8 O1–O2.7 | 无训练；GT pseudo-match + 显式 Kabsch/几何因子分解 | support、depth、K、scale/affine、Sim3、SAM-instance region | solver 与 GT geometry 可行；predicted depth/K 是当前显式 pose 的瓶颈 | E3 诊断结论，不是 SAM-token 结论 |
| V9（Stage O/O-R1） | visible-surface 2D correspondence → fixed epipolar solver；后续才接 true SAM local token | calibrated K、raw SAM3.1 动态 slot、最近两次因果历史、short/medium/long、exact fallback | 初始 solver all-fold fail；O-R1 双初始化/有界融合已实现待运行；不读取 predicted depth/pointmap，不训练 matcher/pose head | E0 |

## 4. 分版本结果

### 4.1 V4：coverage-first 整体系统

已完成：

- 固定实例 `[37, 68, 54]`，frame 90 为 reference；
- 训练帧 `90,105,119,130,140`，同 clip held-out `210,240`；
- 第二 clip `492–589`；
- `aligned` 与 `module_off`；
- SAM `detector_fpn2` pooled appearance、mask/ID、MATCH geometry/ray support、局部 DPT patch 更新；
- pooled appearance/geometry 的 current、memory、difference 被编码为 persistent instance token；
- instance token cross-attend 到 mask 内 StreamVGGT DPT patch token，随后由冻结的 depth head 和
  point head 解码 refined depth/world pointmap；
- 训练损失已经包含 scale-invariant depth、reference-fixed depth、aligned pointmap 和 pose/rigid
  项，因此广义的“用 SAM/实例信息学习 depth shape/pointmap 修正”不是未做实验。

结果边界：V4 因配置序列上 pose 表现较强而被保留，但正式控制只有整个 module 开/关。
`module_off` 只能证明模块整体改变了 depth/pointmap/pose，不能区分 pooled appearance、SAM mask、
persistent ID、StreamVGGT geometry、learned DPT residual、直接 camera 更新或 ray solver 的贡献。
它使用 pooled instance appearance，不是 V7.2 起的逐点 SAM local token；同时 depth、pointmap 和
pose 联合训练，所以也没有证明 depth 改善是 SAM descriptor 而非 geometry/mask 或其它 loss 导致。

因此不得写成“V4 证明 SAM token 改善 pose”。

主要复现入口：

```text
streaming_couping/commands_v4_coverage_first.txt
streaming_couping/configs/v4_coverage_first.yaml
outputs/streaming_couping_v4_coverage_first/evaluation/ray_pose_compact_summary.csv
```

### 4.2 V5：adaptive pointmap 与解析相机中心

已完成：

- 与 V4 相同的固定实例、训练/held-out/第二 clip；
- learned pose、raw pointmap、learned pointmap、固定 blend 与 adaptive support；
- 沿用 V4 的 instance-token → mask-local DPT patch → learned depth/pointmap 修正；
- V5 angular-Huber ray-centre solver；
- `module_off` bit-exact 检查。

结果边界：最终改进同时包含 learned DPT adapter、pointmap 选择、support gate 和解析 solver。
它证明“含 SAM 的完整 pipeline 可改善配置序列”，没有 token-only、geometry-only、wrong-ID 或
shuffle-time 的公平 held-out 因果比较。

主要复现入口：

```text
streaming_couping/commands_v5_adaptive_best.txt
streaming_couping/configs/v5_adaptive_best.yaml
outputs/streaming_couping_v5_adaptive_best/adaptive_upload_summary.csv
```

### 4.3 V6：混合实例输入与直接 SE(3) head

已完成：

- `camera_only`、`instance_only`、`fusion`；
- `instance_off`、`camera_off`、`shuffle_time`、`wrong_geometry`、
  `appearance_only`、`geometry_only`；
- 训练序列过拟合、同 clip held-out、validation clip、第二 clip；
- camera rotation/center、instance rotation/center、fusion 与 specialized 分支。

关键结果：

| Split | raw loss | camera-only | instance-only | fusion |
|---|---:|---:|---:|---:|
| 训练序列 | 0.169429 | 约 0 | 约 0 | 约 0 |
| held-out `210,240` | 0.424380 | 0.278413 | 0.296199 | 0.291904 |
| 第二 clip | 0.498125 | 0.460143 | 0.444963 | 0.441190 |

第二 clip 相对 raw 的 loss 改善分别约为 camera `7.63%`、instance `10.67%`、fusion `11.43%`。
这是已有 SAM/实例输入最明确的 E2 结果之一，但仍不能升级为 E3：

- instance token 混合 pooled appearance、geometry、quality、mask/ID；
- `shuffle_time` 同时打乱 appearance 与 geometry，不是 SAM-only 时间控制；
- `appearance_only`、`wrong_geometry` 在训练帧仍能得到接近零的 loss；
- 可训练 head 直接输出 SE(3)，可以把 feature 当作帧编码记忆。

不要重复：再次训练 V6 式 `camera/instance/fusion + direct pose head` 不能解决归因问题。

### 4.4 原始 V7：结构梯度

架构梯度已经覆盖：

- L0 camera-only；
- pooled monolithic instance vector；
- camera-to-instance cross attention；
- appearance identity Key + geometry Value；
- local32 hierarchical geometry。

关键 held-out loss：

| 分支 | temporal | validation | 第二 clip | long30 |
|---|---:|---:|---:|---:|
| L0 camera-only | 0.277531 | 0.231047 | 0.478159 | 0.450584 |
| identity-Key/geometry-Value | 0.284053 | 0.239678 | 0.551803 | 0.456867 |
| local32 hierarchical | 0.283351 | 0.241264 | 0.553276 | 0.462488 |

两个较明确的 identity/local 分支均未稳定超过 L0。L2 pooled cross-attention 在部分 split
较好，但单个 persistent vector 作为唯一 Key 时 attention 数学上退化，不能作为局部对应证据。

### 4.5 V7.1：冻结 L0 后的实例内容因果实验

已完成：

- Stage A 只训练 `105–255` camera L0；
- 冻结 L0，Stage B 在 `270–345` 训练 residual；
- development `360–405`、future `420–525`、validation `60–80`、第二 clip；
- camera-extra-all、camera-extra-common-gate、gate-only、appearance-pool、geometry-pool、
  decoupled-global、local32；
- wrong geometry、shuffle time、appearance-only、geometry-only 与 exact instance-off。

结果：所有架构 `causal_instance_pass=0`。部分 development 提升可由 camera capacity control
达到或超过；validation 和第二 clip 大多显著差于 frozen L0。

示例：

| 分支 | development gain | future gain | validation gain | second-clip gain |
|---|---:|---:|---:|---:|
| camera-extra-all | +46.83% | +2.72% | -106.42% | -37.39% |
| appearance-pool | +45.91% | +3.27% | -99.16% | -17.77% |
| geometry-pool | +17.54% | -1.80% | -141.15% | -17.78% |
| local32 | +26.80% | +1.74% | -109.91% | -20.12% |

结论：冻结 L0 解决了共同训练混淆，但 residual head 仍直接把 camera/evidence 映射到 SE(3)，
且实例内容没有通过未来/第二 clip 因果标准。

### 4.6 V7.2：真正的 SAM3.1 local token

V7.2 首次缓存并使用 mask 内 `detector_fpn2` token set，而不是 pooled 单向量：

```text
sam_local_features [S,K,P,C]
sam_local_uv       [S,K,P,2]
sam_local_valid    [S,K,P]
```

已经完成 K=`8/16/32` 的：SAM current pool、SAM local match、geometry local match、dual local、
SAM-Key/global-geometry、dual-Key/global+local geometry，以及 local-off、wrong local identity、
shuffle local time、wrong geometry、instance-off、future/validation/第二 clip。

结果：22 个正式行全部 `causal_local_pass=0`。代表性结果：

- `geometry_local_match_k08`：future `+9.75%`，second clip `+3.98%`；
- `dual_local_match_k16`：future `+10.07%`，second clip `-2.20%`；
- `sam_local_match_k08`：future `-0.68%`，second clip `-11.42%`；
- 所有分支在 validation 都明显差于 frozen L0。

结论：保留 SAM local token 没有稳定超过 mask-local StreamVGGT geometry。

### 4.7 V7.3：SAM affinity、geometry Value

V7.3 已经完整实现以下常被重新提出的方案：

```text
SAM local descriptor → correspondence logits
StreamVGGT local geometry → transported Value
current/matched/diff/product → pooled evidence
camera + evidence → direct bounded SE(3) residual
```

控制包含 uniform、geometry-only、SAM-only、SAM+geometry，以及 `sam_off`、`uniform_sam`、
`wrong_sam_identity`、`shuffle_sam_time`、`wrong_local_geometry`、camera capacity 和保留的 V7.2
geometry control。

#### Held-out 结果

K=8 `sam_geometry_transport` 相对自己的 same-K geometry control 在四个 split 都更好，但：

- development gain vs L0：`+40.35%`；
- future：`+14.60%`；
- validation：`-93.63%`；
- second clip：`-29.64%`；
- SAM 破坏测试只在 `11/16` 个组合中让结果变差；
- 没有超过保留的 V7.2 geometry control；
- `causal_sam_pass=0`。

因此这是弱的“同一新架构内部相对优势”，不是可发表的 token 因果结论。

#### 长序列 capacity

`105–525` 的 29 个非 reference 帧已经完整监督。uniform、geometry、SAM、SAM+geometry 和
trained-SAM-off 都能把 active-frame loss 压到接近零。`sam_transport` 正常输入 loss 约
`2.53e-7`，SAM-off 后约 `0.2837`，说明训练输出依赖 SAM 输入；但 uniform/geometry 也能
过拟合，而且没有 held-out 帧。

结论只能是 E1：SAM feature 可以充当训练帧编码，不能推出它学到了跨帧物理对应。

### 4.8 V7.4：动态实例与严格时间 fold

V7.4 已完成“物体不在 frame 90、后续中途出现”的流程：

- SAM3.1 forward-only `bed`/`wardrobe` discovery；
- 最多 8 个 permanent logical slots；
- 任意帧 birth；birth 帧只写 memory；第二次可靠观测才产生 pose evidence；
- 每次只读取最近的早期历史，预测后才写当前帧；
- 无 mature instance 时逐元素等于 L0；
- short/medium/long 三个 train-prefix → future fold；
- parameter-matched `sam_geometry_train_sam_off`。

已观察到 slot 2 在 frame 150 才 birth，因此中途实例不是待实现功能。

主结果：

| Fold | SAM+geometry gain vs L0 | vs geometry | vs trained-SAM-off | 结论 |
|---|---:|---:|---:|---|
| short | +22.81% | -5.58% | -5.86% | 失败 |
| medium | +10.26% | -2.52% | +8.29% | 失败 |
| long | +22.01% | +9.21% | +3.29% | 失败 |

long 虽超过两个控制，但 `sam_off / wrong-ID / shuffle-time` 的 damage 约为
`-0.88% / -0.88% / -0.67%`：破坏 SAM 反而略好。因此三个 `fold_sam_causal_pass` 和
`all_folds_sam_causal_pass` 都为 0。

不要重复：动态 birth、因果 memory、三个未来 fold 和 trained-SAM-off 已经做过。

### 4.9 V7.5：不训练 pose head 的 region/identity 显式约束

V7.5 不读取 SAM local token。它用 SAM mask 内冻结 StreamVGGT point、bounded ICP 和 V5
ray-centre solver，对比 full image、bbox、same-area random、stale mask、current-only、wrong-ID
和 GT-mask oracle，并同时测试 frozen-prefix 与 online memory。

结果：

- frozen-prefix all-fold region pass：`0`；
- online-memory all-fold region pass：`0`；
- frozen-prefix all-fold history pass：`0`；
- online-memory all-fold history pass：`0`；
- short/frozen 的 region 与 history 单独通过；medium/long 未稳定通过；
- long/online 相对 raw 反而下降约 `3.51%`。

结论：SAM region/ID 在局部 fold 有价值，但不足以稳定改善所有未来帧；该实验不能用于声明
SAM descriptor/token 有帮助。

### 4.10 V8 matcher-first：显式匹配监督与冻结匹配

V8 已经排除 V7 的两个主要捷径：

- Stage A 用 GT-world pseudo-match 单独监督 `p_ij`；
- Stage B 冻结 matching，pose head 使用 `evidence_only`，不读取 camera hidden；
- 另有 geometry control、trained-SAM-off、无 match supervision、SAM dual Value；
- short/medium/long 使用相同动态实例 cache。

结果仍未通过：

| Fold | SAM+geometry pose gain vs L0 | pose gain vs geometry | fold causal pass |
|---|---:|---:|---:|
| short | +58.83% | +4.99% | 0 |
| medium | +12.79% | -1.48% | 0 |
| long | -37.78% | -4.76% | 0 |

其它关键信号：

- held-out supervised query 很少：short/medium/long 分别约 `10/2/11`；
- short 的 no-match-supervision pose loss `0.0947`，优于 supervised SAM+geometry 的
  `0.1389`，说明 short pose 提升不能归因于学到的 matcher；
- SAM perturbation 没有在三个 fold 稳定伤害 matching；
- dual SAM Value 仅在 long 相对 geometry Value 改善，short/medium 分别下降约
  `61.54%/10.56%`，且 long 仍差于 L0。

因此 V8 matcher-first 同样没有得到 SAM-token E3 结论。

### 4.11 V8 O1–O2.7：显式几何能力诊断

这组实验不训练模型，使用 GT-world pseudo-correspondence 和 Kabsch/Umeyama，目的是先证明
geometry/solver 上界，避免继续把 geometry 失败误判为 SAM matcher 失败。

#### O1/O2.5：support 与几何来源

- 坐标审计通过；
- `K=64 / history=2 / mutual / 0.10m` 是自动选择的最轻 all-fold reliable support；
- O1 GT camera geometry + GT history 在 weighted/trimmed Kabsch 下通过；
- predicted depth 的 O2-GT-history 与 O2-L0-history 都失败；
- point-head O2P 由于使用 L0 W2C 转 camera，只能作为 pose-entangled 诊断。

结论：对应支持和 solver 已有可行上界，预测 camera geometry 是瓶颈。

#### O2.6：depth、intrinsics、calibration、Sim3 分解

- direct GT-camera O1：通过；
- GT depth + GT K：通过，证明采样和反投影正确；
- GT depth + predicted K：失败；
- predicted depth + GT K：失败；
- predicted depth + predicted K：失败；
- oracle scale/affine depth：仍不能全 fold 通过；
- uncalibrated predicted geometry Sim3：失败。

结论：predicted K 会损害 pose，predicted depth 即使配 GT K 也失败；问题不只是全局 scale、
offset 或刚性/相似变换选择，而是局部 depth shape/跨帧一致性。

#### O2.7：固定 K 与 SAM-instance depth affine

- current/reference/causal-median predicted K 全部未通过；
- global affine depth + GT K 未全 fold 通过；
- SAM-instance affine depth + GT K 也未全 fold 通过；
- 但 instance affine 相对 global affine 的 fold gain 更高：
  short `49.89% vs 34.90%`、medium `27.73% vs 18.99%`、long `90.05% vs 81.11%`；
- medium 的 frame 435 仍是关键失败帧：单实例、约 11 对应、历史 330/345，instance affine
  + GT K 相对 L0 约 `-30.89%`；
- frame 450/465 的同一 oracle 分支分别约 `+34.78%/+75.12%`。

结论：SAM mask 定义的实例区域对 depth calibration 有 E2 上界价值，但 predicted depth/K 和
陈旧/稀疏历史仍阻止稳定显式 pose；这仍不是 SAM token 结论。

## 5. 验证项索引：哪些不能再重复设计

| 常见提议 | 已完成版本 | 当前结论 |
|---|---|---|
| 用 SAM/实例信息学习纠正 depth shape/pointmap | V4、V5 | 已完成；整体 pipeline 为 E2，但 pooled SAM token、mask、geometry、pose head 和 solver 未隔离 |
| mask-local DPT patch residual + depth/pointmap 联合监督 | V4、V5 | 已完成；不得表述为全新物理中间量方案 |
| 同一序列能否过拟合 | V6、V7、V7.3 long、V7.4、V8 | 能；直接 pose head 下没有因果意义 |
| camera-only / instance-only / fusion | V6、V7、V7.1 | 已完成；实例分支有 E2 提升但未归因 token |
| appearance-only / geometry-only | V6、V7、V7.1、V7.2、V7.3、V7.4 | 已完成；geometry 通常更稳定 |
| SAM-off | V7.3、V7.4、V8 | 已完成，包括 parameter-matched trained-off |
| wrong SAM identity | V7.2、V7.3、V7.4、V7.5、V8 | 已完成；未形成 all-fold 因果伤害 |
| shuffle SAM time | V7.2、V7.3、V7.4、V8 | 已完成；未形成 all-fold 因果伤害 |
| uniform attention/SAM | V7.3、V7.4 | 已完成；uniform 也可过拟合 |
| true mask-local SAM token | V7.2、V7.3、V7.4、V8 | 已完成，不是 pooled-token 缺失问题 |
| token 数 K=8/16/32 | V7.2；V7.3 K=8/16 | 已完成；增大 K 未证明 SAM 因果价值 |
| SAM Key + geometry Value | V7.3、V7.4、V8 | 已完成；未通过 all-fold |
| SAM + geometry dual Value | V7.2 dual、V8.1 | 已完成；不稳定 |
| frozen early camera L0 | V7.1–V8 | 已完成 |
| future held-out frames | V6、V7.1–V8 | 已完成，同场景 temporal 结果未通过 SAM 因果标准 |
| 第二 clip | V4–V7.3 | 已完成，但仍是同一 scene，不是跨场景 |
| 90:15:525 长序列 | V7–V8 | 已完成 |
| 物体中途出现/dynamic birth | V7.4、V7.5、V8 | 已完成；birth 后第二次可靠观测开始工作 |
| frozen/online causal memory | V7.4、V7.5、V8 | 已完成 |
| 无实例 exact L0 fallback | V7.1–V8 | 已完成；这不同于人为阈值选择“只接受会提升的帧” |
| 不训练 pose head 的显式 solver | V7.5、V8 O1–O2.7 | 已完成；GT geometry 可行，predicted geometry 不足 |
| full/bbox/random/stale region control | V7.5 | 已完成 |
| GT depth / predicted depth 与 GT K / predicted K 分解 | V8 O2.6/O2.7 | 已完成 |
| scale/affine/Sim3 是否能修复 depth | V8 O2.6/O2.7 | 已完成；不能稳定修复 |
| SAM token → 2D–2D correspondence → epipolar pose | V9 Stage O 初始 solver 已运行且全折失败；O-R1 修正待运行；SAM matcher 尚未实现 | 与 V4–V8 不同，不使用 depth/3D Value/direct pose head；必须先过 oracle 停止门 |

## 6. 已确立、不得反向解释的结论

1. V5/V6 的提升是真实的整体 pipeline 结果，但不能倒推为 SAM local token 的贡献。
2. V4/V5 已经做过 learned mask-local DPT depth/pointmap 修正；不得再把“让 SAM 帮助纠正
   depth shape”本身描述为新方向。尚未完成的是对 pooled/local SAM descriptor、mask 和 geometry
   的严格因果隔离。
3. 训练 loss 接近零不构成 SAM token 证据；uniform、geometry、SAM 和 wrong-input 分支都曾过拟合。
4. SAM local token 不是“尚未实现”：V7.2 起已经真实提取并进入模型。
5. 中途实例不是“尚未实现”：V7.4 起已经支持动态 birth 和因果 memory。
6. V7.3/V7.4 已经做过 SAM affinity → geometry Value；不得再以相同信息流新建版本。
7. V7.4/V8 的 all-fold SAM causal pass 均为 0；不能只选择 long 或单帧正结果声称成功。
8. V7.5 只支持关于 mask/ID 的结论，不能被引用为 descriptor/token 证据。
9. O1 + GT-depth/GT-K 已证明 solver 与反投影可工作；当前显式 pose 的主要瓶颈是 predicted
   depth shape、predicted K 以及稀疏/陈旧实例历史，而不是再加一个更大的 fusion MLP。
10. O2.7 表明 SAM instance region 比全图 depth affine 更有上界价值，但仍没有 all-fold pass。
11. 当前所有 evaluation clip 属于同一 scene；跨场景泛化尚未验证。

## 7. 后续实验准入条件

新实验只有满足至少一项，才算没有重复：

- 去掉直接 pose residual，预测一个可单独用 GT 评估的物理中间量；
- 如果重新研究 depth correction，必须明确标为对 V4/V5 旧假设的因果重验，而不是新方向；
  只有同时改为可独立评分的点级物理输出、去掉 pose head、使用 true SAM local-token 与
  parameter-matched geometry/mask-only 控制，并由固定 solver 消费时才准入；
- 在不读取 GT 的推理条件下，先建立 predicted geometry 的 all-fold 上界；
- 使用真正不同 scene 的数据验证，而不是同 scene 第二 clip；
- 改变 upstream 可训练边界，例如只微调明确的 StreamVGGT depth/K 子模块，并设置 raw、
  geometry-only、SAM-mask-only、SAM-token 条件的参数匹配控制。

以下提议默认拒绝，除非能指出与旧实验的严格差异：

- 再做一个更大的 `camera + SAM + geometry → SE(3)` MLP/Transformer；
- 再做一个 V4/V5 式 instance token → DPT patch residual，却不增加 SAM-token/geometry/mask
  因果控制；
- 再扫一次 token 数、attention heads 或 hidden dimension；
- 再做一遍 SAM-off/wrong-ID/shuffle-time，但不改变物理输出路径；
- 只在训练序列把 pose loss 降到零；
- 只挑选改善帧或用 GT pose error 设计回退阈值；
- 把同场景第二 clip 称作跨场景泛化。

当前唯一已登记但未完成的新实验是
[V9 SAM3.1 local-token 二维极几何位姿因果实验](v90_epipolar_token_causality.md)。它必须先通过
无训练的 oracle 2D solver 上界，失败时不允许继续训练 matcher。

## 8. 更新模板

每次完成新实验后，在本文件追加以下信息：

```text
版本/日期：
问题：
与既有实验唯一差异：
训练帧：
选择帧：
严格 held-out 帧/scene：
主分支：
parameter-matched control：
输入破坏控制：
直接 pose shortcut 是否存在：
主 CSV：
关键数值：
pass 字段：
证据等级：E0/E1/E2/E3
允许的结论：
禁止的结论：
下一步是否满足准入条件：
```

任何 runbook 中的目标描述都不能覆盖本账本中已经记录的实际失败结果；如重新运行改变结论，
必须保留旧结果、注明代码/配置差异，并追加新的 dated entry，而不是直接改写历史。
