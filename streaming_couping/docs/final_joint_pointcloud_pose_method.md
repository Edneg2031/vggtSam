# StreamVGGT + SAM3：V4、V5 与 V6 camera 消融

当前代码保留两条完整方法和一条隔离的 camera-fusion 实验：

| 版本 | 用途 | 核心选择 |
|---|---|---|
| V4 coverage-first | 后续解决错检的研究主线 | 保留关联 mask 覆盖率，使用 appearance + additive pose |
| V5 adaptive-best | 点云安全的最终对照 | residual + bounded SO(3)，按 ray support 选择 raw/learned pointmap |
| V6 camera overfit | 三种结构的五帧公平拟合实验 | camera-only / instance-only / fusion，无点云/solver |

旧结构消融、historical-anchor、current-raw solver、joint BA 和 shared-SE(3) 已从代码删除。
历史实验结论体现在这里的版本选择中，不再保留不可运行的实验实现。

## 1. 公共数据流

```text
RGB 序列 + 参考帧实例 mask
  ├─ StreamVGGT
  │    ├─ aggregator camera/patch tokens
  │    ├─ raw world pointmap
  │    └─ raw camera pose
  └─ SAM3 persistent tracking
       ├─ raw masks
       ├─ geometry registration → MATCH / UNKNOWN / MISMATCH
       └─ causal appearance/geometry memory
                    ↓
          persistent instance tokens
             ┌──────┴──────┐
             │             │
      camera-token fusion  patch-token fusion
             │             │
      refined rotation     refined pointmap
             └──────┬──────┘
          angular-Huber point-to-ray
                    ↓
          refined camera translation
```

SAM3 外观来自冻结 detector backbone 的 `detector_fpn2` 特征。每个实例在 mask 内池化
feature mean/std，并和因果 memory、三维统计及质量分数一起编码为 512 维 persistent
instance token。token 只在可靠观测后更新；开始新 clip 时重新初始化。

StreamVGGT point head 使用 aggregator 的第 `4/11/17/23` 层 patch tokens。四层分别做
实例 cross-attention，再送回冻结 DPT point head。camera branch 只修改 camera token，
最终由冻结 CameraHead 解码。

V4/V5 的所有可学习写入均为门控残差：

```text
output = frozen_token + sigmoid(gate) * zero_initialized_projection(attention)
```

projection 初始为零，因此 `module_off` 和零初始化都能精确恢复冻结 StreamVGGT。

## 2. 实例身份与三套 mask

当前观测与参考帧/历史可靠实例点云做有界 3D registration：

- `MATCH`：允许进入 camera、geometry、pointmap 和 ray solver；
- `UNKNOWN`：几何证据不足；可保留在关联输出，是否进入 camera 由版本权重决定；
- `MISMATCH`：明确空间冲突；不进入学习或解析几何分支。

两个版本都完整导出：

```text
segmentation_masks/       MATCH+UNKNOWN 关联结果，便于观察覆盖率
geometry_trusted_masks/   MATCH-only，实际几何支持
raw_tracking_masks/       未经过身份门控的 SAM3 原始追踪
```

当前的新序列问题属于“外观相似、空间不一致的错检”。V4 暂时优先保留覆盖率，后续应改进
UNKNOWN/MISMATCH 判定，而不是继续收紧阈值造成漏检。

## 3. V4 coverage-first

固定配置：

```text
pose feature             = appearance_only
rotation update          = additive_encoding
unknown camera weight    = 0
patch attention          = dilated union mask
ray solver               = current refined pointmap
```

V4 使用 `v4_match_additive_union` 的已训练权重。关联 mask 仍保留 MATCH+UNKNOWN，但 camera、
geometry 和 ray support 只使用 MATCH。这样能够保留可视化覆盖率，同时避免 UNKNOWN 直接
修改位姿。

已记录结果：

| clip | Raw ATE | V4 ATE | Raw pointmap | V4 pointmap |
|---|---:|---:|---:|---:|
| 90–240 | 0.36879 | 0.20464 | 0.15258 | 0.14037 |
| 492–589 | 0.36079 | 0.33316 | 0.22446 | 0.22644 |

结论：

- 两段序列位姿都改善；
- 90–240 点云改善；
- 492–589 pointmap 轻微退化约 `0.0020 m`；
- 新序列仍可能保留错检，因此 V4 是后续研究主线，不是身份关联已经解决的结论。

运行：

```bash
zsh streaming_couping/commands_v4_coverage_first.txt
```

第一次运行优先复制已有 `v4_match_additive_union` checkpoint；若不存在才训练。独立保存后
不会被 V5 覆盖：

```text
outputs/streaming_couping_v4_coverage_first/
├── checkpoints/decoupled_dual_branch/checkpoint_best.pt
├── evaluation/
├── final_v4_coverage_first/
├── v4_coverage_first_artifact_manifest.txt
└── v4_coverage_first_reproduction_inputs.sha256
```

## 4. V5 adaptive-best

固定学习结构：

```text
pose feature             = residual_only
rotation update          = bounded_so3 (max 5°)
unknown camera weight    = 0.25
patch attention          = dilated union mask
reference pose policy    = preserve reference, accepted center blend 0.5
```

最终 pointmap 只保留两个候选：

```text
P(0) = P_raw
P(1) = P_learned（参考帧和无效像素回退 raw）
```

对 `P(1)` 运行相同 ray solver，计算非参考帧接受率：

```text
support_ratio = fit_accepted / nonreference_frames

support_ratio >= 0.75  → 使用 P(1)
support_ratio <  0.75  → 使用 P(0)
```

这个 gate 不读取 GT。GT 只在运行结束后计算 ATE、rotation 和 pointmap error。

已记录结果：

| clip | support | 选择 | Raw→V5 ATE | Raw→V5 rotation | Raw→V5 pointmap |
|---|---:|---:|---:|---:|---:|
| 90–240 | 6/6 | learned | 0.36879→0.15932 | 2.72281→1.36900 | 0.15258→0.14045 |
| 492–589 | 2/5 | raw | 0.36079→0.35361 | 2.05885→2.01505 | 0.22446→0.22446 |

因此 V5 在可靠序列使用 learned pointmap，在低支持序列保持 raw pointmap；第二段序列的
位姿仍由 learned rotation 和 ray translation 获得小幅提升。

运行：

```bash
zsh streaming_couping/commands_v5_adaptive_best.txt
```

第一次运行优先迁移已有 `v5_residual_so3_union` checkpoint；若不存在，只训练这一种结构。
命令随后重新计算固定 reference pose、adaptive gate、完整导出、位姿图和数量检查。

```text
outputs/streaming_couping_v5_adaptive_best/
├── checkpoints/decoupled_dual_branch/checkpoint_best.pt
├── reference_pose/
├── adaptive_evaluation/
├── adaptive_upload_summary.csv
├── final_adaptive_pointmap_pose/
├── adaptive_artifact_manifest.txt
└── adaptive_reproduction_inputs.sha256
```

## 5. 输出内容

两个版本的每个 clip 都导出：

```text
<clip>/
├── deployable_native/
│   ├── camera_poses.csv
│   ├── camera_poses.npz
│   ├── full_scene.ply
│   └── instance_*.ply
├── segmentation_masks/
├── geometry_trusted_masks/
├── raw_tracking_masks/
├── pointclouds/
├── comparison_gt_world/
│   ├── full_scene/{ground_truth,streamvggt_raw,ours,overlay}.ply
│   ├── instance_*/
│   ├── camera_poses.csv
│   ├── camera_pose_metrics.csv
│   ├── pointcloud_metrics.csv
│   ├── pose_comparison.png
│   └── pose_comparison.pdf
└── camera_centers_overlay_metric_gt_world.ply
```

`deployable_native` 不读取 GT，使用 StreamVGGT native gauge。`comparison_gt_world` 使用一次
固定 reference-frame Sim(3)，只用于开发期公平比较；raw 和 ours 使用同一个对齐。

## 6. GT 使用边界

推理读取：

```text
RGB / frozen feature cache
SAM3 tracking 与 identity state
冻结 StreamVGGT 输出
训练好的 adapter
ray fit accepted 状态
```

训练时 GT 用于 pose/depth/pointmap supervision；评估时 GT 用于指标和 GT-world 可视化。
adaptive gate、SAM3 tracking、身份注册和最终 native 输出均不读取测试帧 GT。

## 7. 当前结论与下一步

- V4 是实例覆盖率优先、两段序列位姿较好的研究主线；
- V5 是 pointmap 不退化的安全对照；
- 当前只有一个场景的两个序列，不能声称跨场景通用；
- 下一步只研究 V4 的错检拒绝：使用更可靠的多帧空间一致性，在不降低 mask recall 的前提下
  区分外观相似但空间错误的 UNKNOWN；
- 不再恢复已失败的 joint BA、shared-SE(3) 或额外结构消融。

## 8. V6：camera fusion 拟合能力与解耦位姿消融

V6 不替换 V4，也不把 V5 的安全回退包装成新方法。它只回答三个更基础的问题：camera
分支、instance 分支和二者融合分别能否拟合五帧，以及 fusion 预测是否真的依赖两种输入。

训练序列固定为：

```text
90 105 119 130 140，reference = 90
```

数据流为：

```text
frozen StreamVGGT camera hidden C
frozen SAM3 mask-pooled appearance A
pose/reprojection geometry G + identity/quality Q
                  ↓
trusted causal persistent-instance memory
                  ↓
camera-to-instance cross-attention
                  ↓
MLP Feature Merger([C, I, C*I, C-I]) = Z
                  ↓
new six-DoF head → Δξ = [ω, ρ]
                  ↓
T_v6 = Exp(Δξ) · T_raw
```

`Z` 是 merger 的新输出，不执行 `C + residual`；但最终相机仍以 SE(3) correction 锚定
原始 StreamVGGT，最后一层零初始化。参考帧和无可用 persistent instance 的帧强制
`Δξ=0`，所以零初始化、`instance_off` 和参考帧都能精确回到 raw pose。

persistent memory 只允许几何 `MATCH` 写入。`UNKNOWN` 当前观测可以读取并和可信历史比较，
但不能更新 memory；临时漏检仍可读取已有 token；显式 `MISMATCH` 不参与融合。这使错检不会
污染长期 token，同时避免一次漏检令 camera branch 完全失去实例上下文。

训练冻结 SAM3 与 StreamVGGT，只用 cache 中的 camera/instance feature。三种模型使用相同
merger/head 容量、随机种子、训练步数和位姿监督，区别仅在有效输入：

```text
camera_only    camera hidden → merger → SE(3) head；所有非参考帧可更新
instance_only  learned query → persistent instances → merger → SE(3) head
fusion         camera query → persistent instances → merger → SE(3) head
```

GT pose 在这里明确作为五帧监督目标；pointmap branch、冻结 CameraHead 和解析式 ray solver
全部关闭。因此实验成功只证明容量、梯度和实现路径正确，不证明 held-out 或跨场景泛化。

三个 checkpoint 训练完成后保持固定，继续读取完整因果上下文并只在 `210 240` 计算指标：

```text
90 105 119 130 140 → train
210 240             → held-out evaluation
```

held-out RGB、camera/instance feature 会进入模型；对应 GT pose 只在预测完成后计算误差，不会
作为模型输入。`heldout_fusion_best=1` 表示 fusion loss 同时低于 raw 和两个独立单分支；它是
当前场景两帧证据，不代表跨场景通用。

现有 `210 240` 结果显示 instance-only 的旋转误差最低，而 camera-only 的相机中心误差和总
loss 最低，feature fusion 没有超过二者。因此 V6.1 不再增加一个 learned fusion，而是在结果
空间锁定下面的可解释组合：

```text
主方案：R = instance-only rotation，C = camera-only camera center
反向对照：R = camera-only rotation，C = instance-only camera center
t = -R @ C，W2C = [R | t]
```

这里组合的是旋转与相机中心，而不是直接拼接两个 W2C 的 translation column，否则会破坏
`t=-RC` 的坐标关系。参考帧逐位复制 raw baseline，两分支 checkpoint 均保持固定。由于
`210 240` 已用于提出该结构，它不能再作为最终证明；锁定结构后只在第二段序列
`492 512 520 545 561 589` 上做验证，且绝不重新训练。该序列的 561 还包含已知错误 mask，
因此同时检验实例旋转分支对错检的鲁棒性。

`cross_clip_decoupled_best=1` 的严格含义是主方案总 loss 同时低于 raw、camera-only、
instance-only、feature fusion 和反向组合。即使为 1，也只说明同一场景的另一 clip 有效；为 0
则说明两帧观察到的分支专长不稳定，应否定当前解耦假设。

独立训练结果写为 `trained_camera_only、trained_instance_only、trained_fusion`。随后只对
fusion checkpoint 做依赖性评估：

```text
instance_off       移除所有实例输入，应精确回到 raw
camera_off         camera hidden 置零，检查 camera token 是否真的参与
shuffle_time       appearance/geometry 跨时间错配，身份门控与 active 数不变
wrong_geometry     只把 geometry 跨时间错配
appearance_only    geometry 置零
geometry_only      appearance 置零
```

交换同一帧 instance slot 顺序没有作为消融，因为 cross-attention 对集合排列近似不变。
`fusion_instance_used=1` 要求 normal loss 同时显著低于 `instance_off` 与 `shuffle_time`；
`fusion_camera_used=1` 要求 normal 显著低于 `camera_off`。不能只靠 normal 拟合得好就
宣称两个来源都参与了融合。

一次运行：

```bash
zsh streaming_couping/commands_v6_camera_overfit.txt
```

最终只需复制短表：

```text
outputs/streaming_couping_v6_camera_overfit/v6_cross_clip_summary.csv
```

完整训练、held-out 和输入依赖消融保留在同目录的 `v6_summary.csv`。

同一命令还打印七行 `v6_frame_diagnostics.csv`：

```text
split, frame, usable_instances, geometry_confidence,
camera_rotation_delta, instance_rotation_delta,
camera_center_delta, instance_center_delta,
v63_rotation_delta, v63_center_delta
```

delta 均为“对应分支误差减 raw 误差”，因此负数表示改善。`usable_instances` 是 persistent
encoder 在该帧实际可读的实例 token 数，`geometry_confidence` 是这些 token 的当前平均几何
置信度。GT 只用于事后计算这些 delta；它们不会进入模型或门控。该表用于判断退化是否和
低几何支持共同出现，不用于针对某一帧挑模型、调阈值或改变最终预测。

### 8.1 V6.2：专门化 rotation/center head

V6.1 的逐帧诊断给出了跨两段序列更稳定的结构规律：camera rotation 在全部七个评估帧改善，
instance rotation 只在三帧改善；camera 与 instance center 则都在全部七帧改善，instance
center 的平均改善更大。退化与当前 geometry confidence 不单调，因此 V6.2 不增加针对某帧的
geometry threshold，而是限制分支职责：

```text
frozen camera token → camera-only merger → 3DoF SO(3) head → R
frozen persistent instance token → instance-only merger → 3DoF center head → C
t = -R @ C，W2C = [R | t]
```

rotation head 只优化旋转损失并逐帧保持 baseline camera center；center head 只优化相机中心
损失并逐帧保持 baseline rotation。最终组合参考帧逐位复制 raw，两条路径都不可能越权修改
另一分量。这与把两个联合 6DoF checkpoint 事后拼接不同：V6.2 从训练目标、head 输出维度到
最终重建都执行相同的可辨识性约束。

命令仍保留原来的三个 6DoF 模型和 V6.1 两种拼接作为公平对照，并额外保存：

```text
v6_checkpoint_specialized_camera_rotation.pt
v6_checkpoint_specialized_instance_center.pt
cross_clip_specialized_cameraR_instanceC
```

第二段 `492...589` 的逐帧 GT 已参与形成 V6.2 假设，所以新 specialized 行只能检查实现与
development-set 行为，不能作为新的 held-out 证明。必须在不再改变结构和超参数的前提下，
用第三段序列验证后才能判断该共同规律是否成立。

V6.2 实验中 rotation 专门化从 `2.799°` 改善到 `2.199°`，但 direct-world center 从 raw
`0.14257` 退化到 `0.16801`，使总 loss 退化。因此职责解耦本身没有被整体否定；需要进一步
区分 center-only 监督失败与世界坐标参数化失败。

### 8.2 V6.3：一次运行完成的参数化与 token-source sweep

V6.3 保留全部既有对照，并在相同 seed、步数和训练帧下训练七个 3DoF component model：

```text
rotation / SO(3):                camera, instance, fusion
center / direct world ΔC:        instance
center / camera-frame Δt:        camera, instance, fusion
```

camera-frame center 不直接预测世界向量，而是：

```text
t_center = t_raw + Δt_camera
C_center = -R_rawᵀ t_center
```

因此 correction 会随 baseline camera rotation 等变地转换到世界中心。随后将三个 rotation
source 与三个 local-center source 形成完整 3×3 组合，并额外保留 direct-world
`cameraR_instanceC`，共十个组合。该 sweep 同时回答：失败来自坐标参数化、token 来源，还是
center-only 监督；不会根据单帧 mask 或 GT 动态选路。

一次命令新增两张短表：

```text
v6_component_sweep.csv  # 7 个 component model 的 train/heldout/cross 误差
v6_v63_sweep.csv        # 10 个合法组合的 train/heldout/cross 指标
```

`development_best=1` 只标记第二段 development clip 上的最低 loss，不会进入模型决策。结构
仍需在第三段预先锁定的序列上验证。

### 8.3 V6.4：center 参数化与旋转辅助监督解耦

旧 instance 6DoF checkpoint 的 center 比严格 3DoF local-center 略好，这可能来自两种不同
因素：SE(3) 参数化的额外自由度，或旋转监督对 instance representation 的正则化。V6.4 因此
固定 instance-only 输入和 full-SE(3) head，只改变训练损失中的旋转权重：

```text
L = 10 * L_center + λ_rot * L_rotation
λ_rot = 0, 0.1, 1, 10
```

`λ=1` 直接复用原 instance-only control；另外训练三个权重。推理时四个模型都丢弃 instance
rotation，只抽取 C，并与同一个 specialized camera rotation 合成。严格 3DoF local-center
作为第五行结构对照。输出为：

```text
v6_v64_aux_sweep.csv
```

若 `λ=0` 的 full-SE(3) center 已优于 3DoF center，收益主要来自参数化；若非零 λ 进一步稳定
改善，才支持“辅助旋转监督有正则化作用”。所有 `development_best` 仍只用于分析。

### 8.4 预锁定第三段验证

停止使用前两段调结构后，第三段按“从 50 开始、每 10 帧采样”预注册，并在训练区间从 90
开始前截止，以避免精确训练图像和 100/110/120 等近邻训练视角泄漏：

```text
50 60 70 80，reference = 50
```

第三段只比较三个预锁定输出：raw、候选 A（camera SO(3) + instance 3DoF local center）和
候选 B（相同 camera SO(3) + `λ_rotation=0.1` instance SE(3) center）。两套 checkpoint 都只在
`90 105 119 130 140` 训练；第三段 GT 只在全部预测完成后计算指标。结果单独写入：

```text
v6_validation_summary.csv
```

为隔离“位姿融合”与“开放词汇检测”两个变量，第三段和训练段采用相同的参考帧实例初始化：
只读取 ScanNet++ GT ID `37/68/54` 在参考帧 50 的 mask，哪个可见就用哪个，不要求三者齐全。
缺失 ID 保留为空槽，不会终止整段；至少一个实例通过后续可见性与 identity gate 时即可建立
persistent memory。若某帧没有任何可用实例，camera rotation 也不会单独生效，整个组合位姿
逐元素保持 StreamVGGT raw，因此零实例时两个候选与 raw bit-exact。这里使用 GT reference
instance mask；后续帧 GT mask 不参与候选选择或位姿修正，仅保留已有的事后诊断 IoU。GT pose
仍只用于最后的误差计算。开放文本自动发现应作为单独的 deployable 初始化消融，不能与当前
候选 A/B 的结构比较混在一起。

该表不会根据第三段结果回头选择 blend、阈值、辅助权重或 token source。

`model_overfit_pass=1` 表示对应独立模型的 loss 下降、旋转、平移和参考帧精确性同时达到
阈值；`fusion_instance_used=1` 与 `fusion_camera_used=1` 分别表示 fusion checkpoint 的输入
消融支持两种 token 确实参与预测。三套模型都能拟合仍只说明容量；跨 clip 的固定 checkpoint
结果才用于判断解耦组合是否稳定，但还不能据此声称跨场景通用。
