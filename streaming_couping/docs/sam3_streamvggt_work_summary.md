# SAM3 + StreamVGGT 实例引导几何与相机位姿优化总结

> 本文总结 V1–V6 的方法演进、实验结果和当前结论。历史 V1–V3 用于说明设计来源；当前仓库正式保留 V4、V5，以及隔离的 V6 camera-fusion 消融。

## 1. 结论先行

当前不能简单地说“编号最大的 V6 就是最好版本”。更准确的判断是：

- **V5 adaptive-best 是目前综合数据和稳定性最好的完整系统**。它同时包含 SAM3 实例跟踪、persistent instance token、StreamVGGT camera/patch token 融合、学习式旋转和 pointmap 修正、解析式相机中心修正，以及不读取 GT 指标的安全回退。需要注意，当前报告实验仍用 GT 实例 ID/参考帧 mask 初始化跟踪，并非完全开放词汇部署。
- **V4 coverage-first 是位姿收益和实例覆盖率更激进的研究版本**。它在两段序列上都改善 ATE，但第二段 pointmap 轻微退化，且仍可能保留外观相似的错误 mask。
- **V6 是机制消融，不是 V5 的后继完整系统**。它证明 camera-only、instance-only 和 fusion 分支都有拟合能力，并能在同场景其他帧/clip 上改善部分指标；但它关闭了 pointmap 分支和 ray solver，第三段预锁定验证又因没有 active persistent instance 而全部回退 raw，因此还不能称为泛化最好的最终方法。
- 当前主要结果都来自 ScanNet++ 的同一场景 `00a231a370`。不同 clip 的固定 checkpoint 测试只能说明**同场景跨时间/跨 clip 泛化**，不能据此声称跨场景通用。

因此，本文单独把 **V5** 作为当前最终完整版本详细说明；V6 保留为“为什么 instance token 可能帮助 camera”的结构验证和下一阶段研究依据。

## 2. 问题与总体思路

StreamVGGT 能从图像流预测相机位姿、深度和世界 pointmap，但长时间运动、低纹理区域和视角变化会造成漂移。SAM3 能持续给出物体 mask 和外观特征，却不直接提供度量位姿。

本工作的核心是把静态物体变成跨帧锚点：SAM3 提供“这是同一个物体及其像素范围”，StreamVGGT 提供相机、射线和三维点。系统把每个可靠物体压缩为因果 persistent instance token，再让 camera token 和物体 token 交互以学习旋转修正，让多层 patch token 和物体 token 交互以修正实例区域 pointmap，最后用修正后的三维点与图像射线解析求解相机中心。这样实例不是直接输出相机位姿，而是为相机和点云提供跨帧稳定的物体级约束。

整体流程为：

```text
RGB 序列
├─ StreamVGGT（冻结）
│  ├─ 最后一层 aggregator camera hidden
│  ├─ aggregator 第 4/11/17/23 层 patch tokens
│  ├─ raw camera pose
│  └─ raw depth / world pointmap
└─ SAM3 tracking / recovery（冻结；当前实验提供参考帧 mask）
   ├─ raw instance masks
   └─ detector_fpn2 feature
          ↓ mask 内 mean/std pooling
外观 + 三维注册统计 + 质量 + 因果 memory
          ↓
512D persistent instance tokens
├─ camera cross-attention → bounded SO(3) 旋转修正
├─ 四层 patch cross-attention → DPT point head → learned pointmap
└─ trusted mask 内 point-to-ray → 相机中心/平移修正
          ↓
无 GT support gate → 选择 raw 或 learned pointmap
```

### 2.1 persistent instance 是什么

它不是直接使用 SAM3 decoder 的某个 object token。当前实现从 SAM3 `detector_fpn2` 特征图中，按每个实例 mask 池化通道均值和标准差，再与当前/历史三维描述、二者残差、跟踪与几何质量、memory age 一起编码成 512 维 token。

参考观测用于初始化实例 memory。之后只有通过三维身份检查并满足质量阈值的 `MATCH` 观测才以 `0.9` momentum 更新 memory；证据不足的 `UNKNOWN` 可以按版本设置参与 camera，但不能污染长期 memory；明确空间冲突的 `MISMATCH` 不参与融合。开始新 clip 时 memory 重新初始化，不跨 clip 偷看未来信息。

身份状态与输出 mask 分开保存：

| 输出 | 内容 | 用途 |
|---|---|---|
| `raw_tracking_masks/` | SAM3 原始追踪 | 检查检测/追踪本身 |
| `segmentation_masks/` | `MATCH + UNKNOWN` | 检查关联覆盖率和漏检 |
| `geometry_trusted_masks/` | `MATCH-only` | 实际 pointmap、三维注册和 ray solver 支持 |

### 2.2 token 怎样进入 StreamVGGT

- **Camera 分支**：取 aggregator 最后一层的 camera hidden，以它为 query、persistent instance tokens 为 key/value 做 cross-attention。冻结 CameraHead 解码前后 hidden 的差值；V5 只把其中的旋转增量以最大 `5°` 的 bounded SO(3) 形式写回 raw pose，不让学习分支直接自由改平移。
- **Pointmap 分支**：取冻结 DPT point head 原本使用的 aggregator 第 `4/11/17/23` 层 token。四层分别在可信实例邻域内做 cross-attention，再一起送入冻结 point head；实例区域外强制恢复 raw 输出。
- **解析平移分支**：在可信实例像素中，用 learned pointmap、相机内参和当前旋转构造点到射线的 angular-Huber 目标，解析优化相机中心；不可靠拟合逐帧回退学习位姿。

V4/V5 的写入都是门控残差：

```text
refined_token = frozen_token
              + sigmoid(gate) * zero_initialized_projection(attention)
```

最后的 projection 从零初始化，因此训练开始、`module_off` 或没有有效实例时都能精确恢复冻结 StreamVGGT。这里使用 residual 的目的不是假设 raw 一定正确，而是保留预训练模型的坐标规范和稳定初值，只学习有界修正。

## 3. V1–V6 版本演进

| 版本 | 主要目标 | 核心实现 | 结果与问题 | 当前状态 |
|---|---|---|---|---|
| V1 | 打通原型 | SAM3 mask、StreamVGGT 点云/位姿、坐标对齐与导出 | 训练、对齐协议、mask 和可视化输出尚未统一，没有可作为最终结论的正式指标 | 历史原型 |
| V2 | 验证 learned pose 有效 | persistent instance → camera token；学习位姿修正 | 历史 clip 上 ATE 和旋转均改善，但仅靠学习分支收益有限 | 历史结果 |
| V3 | 加入可部署解析几何 | learned pose/pointmap + point-to-ray 相机中心修正 | 两段 clip ATE 都改善；错误 mask 会明显污染 pointmap | 历史结果 |
| V4 | 严格身份几何，同时保留 mask recall | `MATCH-only` 几何，appearance-only pose，additive pose，union patch mask | 两段 ATE 都改善；第二段 pointmap 轻微退化，`UNKNOWN` 错检仍可能保留 | 保留的研究主线 |
| V5 | 在跨 clip 波动下保证完整系统安全 | residual-only、bounded SO(3)、固定 reference 策略、0.5 center correction、无 GT support gate | 第一段大幅改善；第二段低支持时 pointmap 精确回退 raw，位姿仍小幅改善 | **当前最终完整版本** |
| V6 | 证明 camera/instance token 的拟合能力并研究职责解耦 | 新 merger + SE(3)/3DoF heads；camera-only、instance-only、fusion 及参数化 sweep | 同场景 held-out/cross-clip 有改善，但 fusion 不稳定胜过单分支；第三段没有有效实例，不能形成新泛化结论 | 隔离机制消融 |

## 4. 各版本实验结果

指标阅读说明：V4/V5 的 `90–240` 使用 causal temporal holdout，训练帧为 `90 105 119 130 140`，ATE/rotation 在 `210 240` 上评估；表中的 `6/6` ray support 则统计整段六个非参考帧。`492–589` 是整段 held-out clip。V6 使用独立的 camera rotation/center loss 协议，因此只能在 V6 表内横向比较。

### 4.1 V1：系统原型

V1 的贡献是打通 SAM3 跟踪、StreamVGGT 输出、实例点云、相机位姿和 GT-world 对齐。这个阶段的评估协议和输出定义仍在变化，因此不列入 V2–V5 的数值横向比较，也不补造缺失指标。

### 4.2 V2：learned pose control

历史 clip：`105 109 113 122 254`。

| 指标 | Raw | V2 | 变化 |
|---|---:|---:|---:|
| ATE RMSE | 0.14141 | 0.11942 | -15.6% |
| 平移均值 | 0.10881 | 0.08778 | 改善 |
| 旋转均值 | 0.62581° | 0.53217° | -15.0% |
| RPE translation RMSE | 0.12200 | 0.11646 | 改善 |

这说明学习式 camera 分支确实具备修正能力，但还没有把实例三维点转成强解析约束。

### 4.3 V3：learned pointmap + ray-refined pose

同一历史 clip：

| 指标 | Raw | V2 | V3 |
|---|---:|---:|---:|
| ATE RMSE | 0.14141 | 0.11942 | **0.07714** |
| RPE translation RMSE | 0.12200 | 0.11646 | **0.05758** |
| pointmap RMSE | 0.14019 | — | **0.12926** |

V3 相比 raw 的 ATE 降低约 `45.5%`，说明 learned rotation 与解析式 center/translation 修正可以互补。

新 clip `492 512 520 545 561 589`：

| 指标 | Raw | V2 | V3 |
|---|---:|---:|---:|
| ATE RMSE | 0.36079 | 0.34785 | **0.33008** |
| 旋转均值 | 2.05885° | 1.81362° | 1.81362° |
| pointmap RMSE | **0.26021** | — | 0.32318 |

这段上 ATE 改善约 `8.5%`，但 pointmap 恶化约 `24.2%`。帧 561 已观察到错误 mask，说明仅靠外观追踪会把错误实例区域送入 pointmap 与 solver；这直接推动了 V4 的三维身份门控。

### 4.4 V4：coverage-first

固定结构：

```text
pose feature          = appearance_only
rotation update       = additive_encoding
UNKNOWN camera weight = 0
patch support         = dilated union mask
ray solver            = current learned pointmap
geometry/ray support  = MATCH-only
```

| clip | Raw ATE | V4 ATE | ATE 变化 | Raw pointmap | V4 pointmap | pointmap 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 90–240 | 0.36879 | **0.20464** | -44.5% | 0.15258 | **0.14037** | -8.0% |
| 492–589 | 0.36079 | **0.33316** | -7.7% | **0.22446** | 0.22644 | +0.9% |

V4 的优点是两段 clip 的位姿都改善，同时不因证据不足就删除关联 mask，漏检情况比过严阈值少。缺点是 `segmentation_masks` 仍保留 `UNKNOWN`，错误关联在可视化中可能存在；第二段 pointmap 也有轻微退化。因此 V4 适合作为继续研究“错检拒绝但不牺牲召回率”的主线，不是当前最稳妥的交付结果。

### 4.5 V5：adaptive-best

| clip | ray support | pointmap 选择 | Raw→V5 ATE | Raw→V5 rotation | Raw→V5 pointmap |
|---|---:|---|---:|---:|---:|
| 90–240 | 6/6 = 1.00 | learned | 0.36879 → **0.15932** | 2.72281° → **1.36900°** | 0.15258 → **0.14045** |
| 492–589 | 2/5 = 0.40 | raw | 0.36079 → **0.35361** | 2.05885° → **2.01505°** | 0.22446 → **0.22446** |

第一段 ATE 降低约 `56.8%`、旋转降低约 `49.7%`、pointmap 改善约 `8.0%`。第二段的实例射线支持不足，系统不使用退化的 learned pointmap；pointmap 与 raw 完全一致，同时 ATE 仍改善约 `2.0%`、旋转改善约 `2.1%`。

V5 并非每个单项都胜过 V4：第二段的 V4 ATE `0.33316` 优于 V5 的 `0.35361`；但 V5 在第一段 ATE/rotation 更好，在第二段避免了 pointmap 退化，且两段 rotation 都优于 raw。按“位姿收益 + pointmap 安全性 + 无 GT 指标回退”综合判断，V5 是当前最稳妥的完整版本，而不是所有表格单元格的绝对最优。这个结论仍限于一个场景的两个 clip。

### 4.6 V6：camera fusion 与职责解耦消融

V6 固定在 `90 105 119 130 140` 训练，只使用冻结 cache 和 GT pose supervision；不训练 pointmap，也不运行 ray solver。V6 的 rotation/center/loss 采用独立 camera 实验协议，不能把数值直接当作 V4/V5 的 ATE。当前 V6 门控会把“已观测但零几何对应”的缓存 `MISMATCH` 仅在当前帧软化为权重 `0.25` 的 `UNKNOWN`，但仍只允许 `MATCH` 写入 persistent memory；V4/V5 保持不变。

训练集上 camera-only、instance-only、fusion 都接近零误差，只证明三个结构都有足够容量，不能证明实例因果性或泛化。

固定 checkpoint 在 `210 240` 上：

| 模型 | rotation | center | loss | loss 相对 raw |
|---|---:|---:|---:|---:|
| raw | 2.76988° | 0.11692 | 0.42799 | — |
| camera-only | 1.37899° | **0.07961** | **0.27863** | -34.9% |
| instance-only | **1.31355°** | 0.07992 | 0.28072 | -34.4% |
| fusion | 1.35545° | 0.07989 | 0.27985 | -34.6% |

这两帧说明 instance token 有可迁移信息，但 fusion 没有稳定超过两个单分支：instance-only 的旋转最好，camera-only 的 center 和总 loss 最好。

固定 checkpoint 在跨 clip `492 512 520 545 561 589` 上：

| 模型 | rotation | center | loss | 相对 raw |
|---|---:|---:|---:|---:|
| raw | 2.79918° | 0.14257 | 0.60273 | — |
| camera-only | 2.32510° | 0.11341 | 0.47680 | 改善 |
| instance-only | 3.64991° | **0.10504** | 0.43492 | loss 改善、旋转退化 |
| feature fusion | 2.48719° | 0.11077 | 0.46273 | 改善 |
| camera-R + instance-C | **2.32510°** | **0.10504** | **0.43436** | 当前解耦组合最好 |

后续参数化实验中，full-SE(3) instance center 加 `0.1` 辅助旋转权重得到 center `0.10471`、loss `0.43345`，略优于严格 3DoF center，但该 clip 已参与结构设计，只能视为 development evidence。

预锁定第三段 `50 60 70 80` 的 raw、候选 A 和候选 B 完全相同：

| variant | rotation | center | loss | active frames |
|---|---:|---:|---:|---:|
| raw | 1.97178° | 0.08381 | 0.32571 | 0 |
| candidate A | 1.97178° | 0.08381 | 0.32571 | 0 |
| candidate B | 1.97178° | 0.08381 | 0.32571 | 0 |

原因是该段没有建立可用的 persistent instance，两个候选按设计逐帧严格回退 raw。该段也使用 GT reference instance mask 初始化，但没有利用后续帧 GT mask 或 GT pose 产生修正。这个结果验证了安全回退，却既不能证明也不能否定 token fusion 在第三段的泛化。因此 V6 暂时不能替代 V5。

## 5. 当前最终完整版本：V5 adaptive-best

### 5.1 输入与初始化

V5 推理读取 RGB、SAM3 跟踪/恢复结果、冻结 StreamVGGT 输出、训练好的 adapter 和 ray-fit 状态。当前正式配置的 `instance_source` 是 `configured_gt_reference`：用指定 GT instance ID `37/68/54` 在参考帧的 mask 初始化实例槽。它不要求三个实例都持续可见，后续只要至少一个实例具有可信历史和当前有效观测，就能给对应分支提供约束。

代码也实现了 `sam3_reference`，可在参考帧通过 SAM3 的 `object` 查询自动填充实例槽，但本文表中的 V4/V5 数值不是在该初始化条件下获得的。因此当前结论应称为“给定参考实例的跟踪与几何优化”，不能直接写成完全无 GT 的开放词汇系统。

### 5.2 实例 token 构造

对每个实例和每一帧，系统组合：

```text
current feature
memory feature
current - memory
tracking / geometry / static quality
memory age
```

V5 的 pose branch 使用 `residual_only`：主要让 camera 学习当前几何与持久几何之间的变化，而不是仅凭相似外观直接改位姿。`MATCH` 正常参与并更新 memory；`UNKNOWN` 以 `0.25` 权重进入 camera 以减轻漏检，但不写入 memory；`MISMATCH` 被拒绝。pointmap 和 ray solver 始终使用更严格的 `MATCH-only` mask。

### 5.3 旋转、点云与平移的职责分离

1. Camera hidden 查询实例 token，冻结 CameraHead 把 hidden 变化解码成旋转增量。
2. 旋转以 bounded SO(3) 形式左乘 raw rotation，单帧最大改动 `5°`；学习分支不自由改 camera center。
3. 第 `4/11/17/23` 层 patch tokens 分别查询实例 token，冻结 DPT head 融合四层后输出 learned pointmap；可信实例邻域外恢复 raw pointmap。
4. angular-Huber point-to-ray solver 在可信实例区域解析估计 camera center，接受后应用 `0.5` 的 center correction；参考相机按固定 reference 策略锚定，拒绝的帧保留学习位姿。

这种分工避免 camera head、point head 和解析 solver 同时无约束地修改同一自由度：学习分支主要负责旋转和局部点云，解析几何负责相机中心。

### 5.4 无 GT 自适应 pointmap 选择

V5 最终只比较两个预定义候选：

```text
P_raw     = frozen StreamVGGT pointmap
P_learned = learned pointmap，参考帧/无效像素保持 raw

support_ratio = accepted ray fits / non-reference frames

support_ratio >= 0.75 → 输出 P_learned
support_ratio <  0.75 → 输出 P_raw
```

gate 只读取 ray solver 是否接受，不读取 GT、ATE 或 pointmap GT error。它解决的是“当前 clip 的实例几何支持是否足以信任 learned pointmap”，不是事后按 GT 挑最优结果。

### 5.5 为什么仍然需要训练

SAM3 mask 和三维注册只能说明物体身份、可见区域与几何可信度，不能直接给出 camera hidden 应如何变化，也不能学习 StreamVGGT 多层 patch feature 应如何在实例区域修正。训练用于学习“camera/patch token 与 persistent instance token 的关系如何映射成有界残差”；ray solver 则提供可解释的解析 center 修正和验收条件。两者职责不同，因此解析式 solver 的存在并不能替代训练。

### 5.6 GT 使用边界

| 阶段 | 是否使用 GT | 用途 |
|---|---|---|
| 训练 | 是 | pose、relative rotation、translation direction、depth、pointmap 等监督 |
| 参考帧初始化 | **当前实验使用** | `configured_gt_reference` 提供实例 ID 和参考帧 mask；这是当前部署性限制 |
| 参考帧之后 | 不用 GT 产生修正 | SAM3 tracking/recovery、三维 identity、adapter 和 ray solver 不按后续 GT mask/pose 选择结果；后续 GT mask 只可计算诊断 IoU |
| adaptive gate | 否 | 不读 GT pose/pointmap 指标，只按 accepted ray-fit ratio 选择 raw/learned pointmap |
| 评估与可视化 | 是 | 计算 ATE、rotation、pointmap 指标，并做统一 GT-world 对齐 |

`deployable_native/` 表示几何结果保持在 StreamVGGT 自身 gauge 中，导出坐标不需要 GT 对齐；它不代表当前整条运行已经摆脱 GT reference mask。`comparison_gt_world/` 只用于公平评估，raw 与 ours 使用同一个固定 reference-frame Sim(3) 对齐。

## 6. 当前泛化结论与限制

现有证据支持以下表述：

- V2/V3 证明实例引导学习分支和解析 ray solver 均有改善位姿的能力。
- V4 证明严格三维身份门控后，两段同场景 clip 的位姿都可改善，但 mask 错检和 pointmap 退化尚未完全解决。
- V5 在高支持 clip 使用 learned pointmap 获得明显收益，在低支持 clip 自动回退 raw pointmap，表现出当前最好的**同场景跨 clip 稳定性**。
- V6 证明 camera token 与 instance token 分别具有拟合和一定迁移能力，但当前 feature fusion 没有稳定超过单分支，预锁定第三段也没有产生有效实例约束。

尚不能声称：

- 对未见过的 ScanNet++ scene 或真实采集场景普遍有效；
- V6 的 instance 分支一定在新场景改善旋转；
- 当前三维身份门控已经彻底解决相似外观错检；
- V5 gate 能判断所有类型的失败，它只防止低 ray support 时的 pointmap 退化。
- 当前表格已经证明无需 GT reference mask 的开放词汇部署。

下一步最有价值的验证是冻结 V5 的结构、权重和阈值，在多个未参与开发的新场景上一次性测试，并同时报告 mask 身份准确率、ray support、逐帧位姿和 pointmap 指标。只有这样才能把“同场景跨 clip 泛化”提升为“跨场景泛化”。

## 7. 复现入口

```bash
# V4 coverage-first
zsh streaming_couping/commands_v4_coverage_first.txt

# 当前最终完整版本 V5
zsh streaming_couping/commands_v5_adaptive_best.txt

# V6 camera/instance/fusion 机制消融
zsh streaming_couping/commands_v6_camera_overfit.txt
```

V4/V5 命令在当前服务器优先调用 `3am/bin/python`，避免从 `(horizonstream)` shell 误用
不同 PyTorch 版本重放冻结 DPT head。其他机器可设置 `STREAMING_COUPING_PYTHON` 覆盖路径。

V5 的主要结果：

```text
outputs/streaming_couping_v5_adaptive_best/
├── checkpoints/decoupled_dual_branch/checkpoint_best.pt
├── adaptive_upload_summary.csv
├── adaptive_evaluation/
└── final_adaptive_pointmap_pose/
    └── <clip>/
        ├── deployable_native/
        ├── segmentation_masks/
        ├── geometry_trusted_masks/
        ├── raw_tracking_masks/
        ├── pointclouds/
        ├── comparison_gt_world/
        └── comparison_gt_objects/
```

`comparison_gt_objects/instance_*/gt_object.ply` 使用 GT instance mask，是真正的 GT 可见物体
点云；`ours_predicted_object.ply` 使用方法实际消费的几何 tracking mask（V5 为 strict
geometry trusted mask），包含分割与几何两类误差；
`ours_on_gt_mask.ply` 固定 GT mask，只检查几何。`object_comparison_metrics.csv` 用一行汇总
每个实例的 mask IoU/precision/recall 和 GT-mask 内的 raw/ours 3D 误差。
命令行打印更短的 `object_comparison_short.csv`，便于直接复制结果。
参考帧使用 GT mask 初始化，因此分割结果以短表的 `nonref_mask_*` 为准；GT PLY 表示所选
帧内可见 GT 深度点，不等同于完整场景 mesh。

## 8. 最终推荐

- 需要当前可复现的完整结果、点云安全性和不依赖 GT 指标的自适应逻辑：使用 **V5 adaptive-best**；当前仍需要给定参考实例。
- 需要继续研究相似外观错检、希望优先保留实例召回率：以 **V4 coverage-first** 为研究基线。
- 需要研究 instance token 是否真正帮助 camera、旋转与 center 应由哪个 token source 负责：使用 **V6**，但不要把它作为完整系统最终指标。
