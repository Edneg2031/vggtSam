# 实例引导的点云与相机位姿优化

## 0. 当前实验：pointmap residual blend

下面第 1–8 节描述的是生成候选结果的冻结 StreamVGGT、SAM3 tracking 和 V5
trainable adapter。它们原先分别输出 learned pose、learned pointmap，再用独立 ray solver
修改相机中心。

第一版 joint BA 曾把 raw/learned pointmap 转成 camera-ray depth，再使用优化后的 pose
重新生成世界点云。实验否决了这个设计：原序列 pointmap mean 从 `0.1526` 恶化到
`0.5087`，新序列从 `0.2245` 恶化到 `0.2554`；深度混合系数仅约 `0.016`，说明破坏主要
来自用不够准确的 pose 强制重新放置整张 raw 几何，而不是 learned depth。

随后测试的 shared-SE(3) graph 能安全回退，但所有帧都因 3D 残差没有可靠下降而拒绝校正。
而且新序列的 learned pointmap 内部残差优于 raw（`0.1455 < 0.1488`），GT error 却更差，
证明当前 patch matching 的内部一致性不能作为继续放宽 BA 的依据。

当前版本保留相同的 learned rotation 和 reference-preserving ray-center solver，只对 point
head 的世界坐标 residual 做确定性缩放：

```text
P(alpha) = P_raw + alpha * (P_learned - P_raw)
             ↓
对每个 alpha 使用 P(alpha) 重新运行同一 fixed_ref_050 ray solver
             ↓
输出配对的 pose(alpha) 与 pointmap(alpha)
```

`alpha ∈ {0, 0.25, 0.5, 0.75, 1}`。参考 pointmap 始终精确取 raw，参考 pose 仍由 solver
固定；无效 learned pixels 也回退 raw。原 temporal holdout 用于选择 alpha，492–589 只报告
选定系数的泛化结果。运行时不读取 GT；GT 只在完成后计算指标。一次运行比较：

```text
raw_control                 原始 StreamVGGT
fixed_ref_050_control       旧 alpha=1 结果复现控制
fixed_pose_pointblend_a000  raw pointmap
fixed_pose_pointblend_a025
fixed_pose_pointblend_a050
fixed_pose_pointblend_a075
fixed_pose_pointblend_a100  learned pointmap
```

运行：

```bash
zsh streaming_couping/commands_joint_pointmap_ba.txt
```

精简结果：

```text
outputs/streaming_couping_v5_ablation/joint_ba_upload_summary.csv
```

## 1. 方法概览

目标是从一段 RGB 视角序列同时得到：

- 跨帧一致的实例分割 mask；
- 修正后的 StreamVGGT world pointmap；
- 修正后的相机旋转和平移。

完整数据流为：

```text
RGB 序列 + 参考帧实例 mask
  │
  ├─ StreamVGGT → 初始 pointmap、camera pose、aggregator tokens
  │
  └─ SAM3 tracking
       └─ 几何不一致时，用历史实例点云筛选 recovery mask 并写回 memory
                  ↓
       当前 mask 点云与因果实例点云做 bounded 3D registration
       不通过：隔离 token / memory / pointmap / ray
                  ↓
          persistent instance observations
                  ↓
       ┌──────────┴──────────┐
       │                     │
四层 patch-token 融合   camera-token 融合
       │                     │
frozen DPT point head   frozen CameraHead
       │                     │
refined pointmap        refined rotation
       └──────────┬──────────┘
                  │
      实例区域 angular-Huber point-to-ray
                  │
          refined camera translation
```

最终无 GT 输出是同一 StreamVGGT native gauge 中的：

```text
persistent instance masks + refined pointmap + refined camera pose
```

## 2. SAM3 tracking 与几何恢复

参考帧 mask 来自选定的 ScanNet++ instance ID。SAM3 首先以该 mask 初始化 persistent
object ID，并按配置中的视角顺序追踪。帧号不要求递增，列表顺序就是模型输入顺序。

系统同时把可靠 mask 内的 StreamVGGT 三维点累计为该实例的历史 object map。当当前 tracker
mask 与 object map 的投影明显不一致时：

```text
tracker geometry coverage < 0.50
  → SAM3 在当前帧重新生成完整候选 masks
  → 用投影几何 support 排序
  → selected support coverage ≥ 0.25
  → 将候选写回同一个 object ID 的 SAM3 memory
  → 从该位置继续追踪后续输入视角
```

每段序列最多接受第一个自然恢复事件。没有触发、候选为空或 support 不足时，直接保留原始
SAM3 tracking。候选选择不使用后续帧 GT。

V4 不再把 recovery 的单向投影覆盖率当成最终身份判定。无论 SAM3 是否触发 recovery，
每帧每个 mask 内的当前三维点都必须与由参考帧和历史可靠观测构成的实例点云完成有界
translation-only registration。注册要求足够多的对应点、足够高的 fitness、足够低的 RMSE，
且中心平移不超过上限。这个检验使用 StreamVGGT 当前 pointmap，不读取当前帧 GT。

因此，一个外观类似但位于错误空间位置的 mask 即使 SAM3 分数很高，也会被标记为
`GEO-REJECT`。

## 3. Persistent instance token

### 3.1 SAM3 外观特征来自哪里

mask 由 SAM3 video tracker 产生，外观特征则来自冻结的 SAM3 image detector backbone：

```text
source = detector_fpn2
tensor = backbone_out["backbone_fpn"][-1]
text conditioning = none
feature grid = 72 × 72
```

它不是 tracker memory token，也不是最终 mask-decoder token。对每帧每个实例，将 tracking
mask resize 到该特征层，然后池化：

```text
appearance_t = [mask 内 FPN feature mean, mask 内 FPN feature std]
```

### 3.2 为什么叫 persistent

同一个 object ID 在当前 clip 内维护 appearance 和 geometry 两份因果 EMA memory：

```text
memory_t = 0.9 × memory_(t-1) + 0.1 × current_t
```

当前实例表示包含：

- 当前、历史及二者差值的 SAM3 appearance；
- 当前、历史及二者差值的三维中心、协方差、extent、ICP 和点云质量；
- tracker、geometry、static confidence；
- memory age。

这些量经过 MLP 编码成 `512` 维 persistent instance token：

```text
instance_token_t =
    MLP(current_t, memory_(t-1), current_t - memory_(t-1), quality_t, age)
```

因此它不等于一个 mask-pooled SAM3 token，而是“当前实例证据 + 同一实例历史”的对象级表示。
memory 在每个可靠观测后更新，开始一个新 clip 时重新初始化。V4 将“本帧临时使用”和
“写入长期 memory”分开：

```text
本帧 token 可用 =
    SAM3 score 通过 && bounded 3D identity registration 通过

memory 可写 =
    本帧 token 可用 && geometry confidence 通过 && static score 通过
```

第一次可靠观测只建立 memory，不修改 StreamVGGT；从下一个可靠观测开始 token 才能参与
融合。当前 mask 不可靠时，该帧不写 memory，也不修改 StreamVGGT。

点云和相机使用两个独立 tokenizer：

- geometry tokenizer 使用 appearance、geometry 和质量特征，并经过统一的 3D identity
  硬门控；
- pose tokenizer 将 geometry 输入置零，只使用 persistent SAM3 appearance、tracker
  confidence 和 memory；它不读取几何描述符，但仍必须通过外部 3D identity 硬门控。

## 4. 实例如何修正 pointmap

### 4.1 使用哪些 StreamVGGT tokens

冻结 StreamVGGT aggregator 先完整运行。系统取其 DPT point/depth head 原本使用的四层输出：

```text
aggregated_tokens_list[4]
aggregated_tokens_list[11]
aggregated_tokens_list[17]
aggregated_tokens_list[23]
```

每层包含 camera/register prefix 和 image patch tokens。只更新
`patch_start_idx` 之后的 patch 部分，prefix 完全保持原值。

### 4.2 四层分别做 cross-attention

四层各有独立的 8-head cross-attention、projection 和 gate。对每一层：

```text
query = 该层 StreamVGGT patch tokens
key/value = 当前帧有效 persistent instance tokens

update_l = CrossAttention_l(query, key, value)
patch_l' = patch_l + sigmoid(gate_l) × zero_proj_l(update_l)
```

V4 将上述残差限制在通过 3D 身份验证的实例 mask 对应 patch（外扩一个 patch）内。DPT
解码后再进行一次像素级回退：验证区域外的 pointmap、depth 和 confidence 直接取原始
StreamVGGT 输出。因此错误 mask 被拒绝时不会修改该区域，也不会让全局 patch 残差污染
背景。

这一步发生在 frozen aggregator 运行之后、DPT head 融合之前。四层被独立修正，再一起送进
冻结的 DPT point/depth head，由该 head 完成多层融合并输出 refined world pointmap、depth
和 confidence。

当前实现不是在 aggregator 的每个 block 内注入：layer 4 的修改不会继续传播到
layer 5–23。准确说法是：

```text
post-aggregator multi-level DPT token conditioning
```

### 4.3 什么是 zero-initialized gated residual

`zero_proj` 权重初始化为零，`gate_l` 是每层一个可学习标量。初始化时：

```text
zero_proj(update) = 0
patch_l' = patch_l
```

因此未训练模块逐元素等价于原始 StreamVGGT。训练后 projection 和 gate 才学会写入小残差；
没有有效实例时，由于 projection 没有 bias，输出仍严格不变。

## 5. 相机旋转如何修正

相机分支使用 frozen aggregator 最后一层的第一个 token：

```text
camera_hidden = aggregated_tokens_list[-1][:, :, 0]
```

每帧只有一个 camera query：

```text
query = camera hidden token
key/value = appearance-only persistent instance tokens
camera_hidden' =
    camera_hidden + sigmoid(gate) × zero_proj(cross_attention)
```

修正后的 hidden token 按 StreamVGGT 原始 causal KV-cache 路径送进冻结 CameraHead。系统同时
解码 raw 和 refined hidden，只把二者的 pose delta 加到精确缓存的 baseline pose encoding，
从而保证关闭模块时严格恢复原始输出。

CameraHead 实际预测完整的：

```text
translation + quaternion rotation + FOV/intrinsics
```

learned 分支并非结构上只输出旋转。但最终方法只保留它改善后的 rotation；learned camera
center 作为几何求解的初值/失败回退，learned intrinsics 被舍弃。

## 6. 相机平移如何修正

最终内参固定为原始 StreamVGGT 在参考帧预测的 `K`，并在整段序列复用。对每帧、每个实例
mask 内的高置信 pointmap 像素：

- refined pointmap 给出世界点 `X`；
- refined rotation 和 reference `K` 给出世界射线方向 `r`；
- 正确相机中心 `C` 应使 `X-C` 与 `r` 共线。

系统在三个 tracked-instance masks 的并集上最小化：

```text
e_line(C)  = (I - rrᵀ)(X - C)
e_angle(C) = ||e_line(C)|| / ||X - C||

min_C Σ w × Huber(e_angle(C))
```

这是每帧只有三个未知量的 angular-Huber IRLS，不需要训练。当前配置：

- minimum points：`1024`；
- maximum iterations：`6`；
- confidence threshold：`0.30`；
- condition-number gate：`1e8`；
- point-to-ray RMSE gate：`0.20` native units；
- center-shift gate：`0.75` native units。

通过检查后，用 solved center 与 refined rotation 重新组成最终外参；失败时回退 learned
camera center。

## 7. 哪些部分需要训练

SAM3 和 StreamVGGT 主干全部冻结。只训练：

- pose/geometry 两个 instance tokenizer；
- 五个 cross-attention：四个 patch 层和一个 camera 分支；
- 对应的 zero projection 和 scalar gate。

需要训练是因为“实例证据应怎样修改 patch/camera token”没有闭式公式。训练损失包括
pointmap、depth、camera encoding、相对旋转、平移方向和实例刚体/质心一致性。

训练一次后，新序列使用固定 checkpoint：

| 模块 | 原训练序列 | 新测试序列 |
|---|---|---|
| SAM3 / StreamVGGT | 冻结 | 冻结 |
| learned adapter | 训练 | 固定，不训练 |
| tracking 与 persistent memory | 重新计算 | 重新计算 |
| ray translation | 逐帧求解 | 逐帧求解 |
| GT | loss 与评估 | 仅评估 |

## 8. 当前结果

### 8.1 原序列 temporal holdout

训练帧为 `90 105 119 130 140`，held-out 为 `210 240`。

| 指标 | Raw StreamVGGT | Learned adapter | Final |
|---|---:|---:|---:|
| pointmap mean error | 0.15258 m | 0.06358 m | 0.06358 m |
| ATE RMSE | 0.36879 m | 0.37535 m | **0.14387 m** |
| translation RPE | 0.23981 m | 0.17627 m | **0.08986 m** |
| rotation RPE | 3.518° | **1.171°** | **1.171°** |

这里 learned adapter 明显改善 pointmap 和 rotation，ray solver 显著改善 translation。

### 8.2 固定权重的新视角序列

实际输入帧为 `105 109 113 122 254`；`100_500` 只是旧 clip 标签。

| 位姿指标 | Raw | Learned | Final |
|---|---:|---:|---:|
| ATE RMSE | 0.14141 m | 0.11942 m | **0.07714 m** |
| translation mean | 0.10881 m | 0.08778 m | **0.06756 m** |
| translation RPE | 0.12200 m | 0.11646 m | **0.05758 m** |
| rotation mean | 0.6258° | **0.5322°** | **0.5322°** |

5/5 ray fits accepted，point-to-ray RMSE 从 `0.04821` 降至 `0.01277` native units。

点云泛化不稳定：

| pointmap mean error | Raw | Ours |
|---|---:|---:|
| full scene | 0.09568 m | 0.09364 m |
| instance 37 / cabinet | **0.05516 m** | 0.08597 m |
| instance 68 / wardrobe | **0.05278 m** | 0.09898 m |
| instance 54 / bed | 0.18653 m | **0.14676 m** |

因此当前最可靠的结论是：

- learned rotation 和 analytic translation 已表现出跨视角价值；
- learned pointmap 在原训练视角有效，但固定权重在新视角上有实例偏置；
- 下一步应为 pointmap residual 增加无 GT 几何可靠性门控，不可靠时回退 raw pointmap。

该新序列包含原训练帧 `105`，所以不是严格无重叠泛化实验。

## 9. 输出

### 9.1 无 GT 可部署结果

```text
final_instance_ray_pose_v4/<clip>/
  deployable_native/
    full_scene.ply
    instance_<id>.ply
    camera_poses.csv
    camera_poses.npz
  segmentation_masks/
    sequence_overview.png
    overlays/
    binary/instance_<id>/
    binary/union/
    mask_summary.csv
    identity_gate_diagnostics.csv
  raw_tracking_masks/
    sequence_overview.png
    overlays/
    binary/instance_<id>/
```

上述点云和位姿使用同一个 StreamVGGT native gauge。`overlays/` 是 RGB 与彩色实例 mask，
`segmentation_masks/binary/` 是严格方法实际消费的 0/255 masks；被几何拒绝但便于诊断的
SAM3 原始结果保存在 `raw_tracking_masks/`。

### 9.2 GT / raw / ours 公平比较

```text
final_instance_ray_pose_v4/<clip>/comparison_gt_world/
  full_scene/{ground_truth,streamvggt_raw,ours,overlay}.ply
  instance_<id>/
  pointcloud_metrics.csv
  camera_pose_metrics.csv
  camera_poses.csv
  pose_comparison.png
  pose_comparison.pdf
```

只从 raw StreamVGGT 参考帧 pointmap 拟合一次 Sim(3)，同一个变换应用于 raw 和 ours，禁止
为 ours 单独重新对齐。比较目录属于 evaluation-only；deployable output 不使用 GT 对齐。

## 10. 运行

重新训练严格几何版，并用固定 checkpoint 测试 `492 512 520 545 561 589`：

```bash
zsh streaming_couping/commands_strict_geometry_train_then_test_492_589.txt
```

使用固定 checkpoint 测试另一输入顺序：

```bash
zsh streaming_couping/commands_final_joint_pointcloud_pose_test.txt
```

已有 cache 和 ray 结果、只补导出：

```bash
PYTHONPATH=src:. python -m streaming_couping.scripts.run_instance_token_pose \
  --config streaming_couping/configs/final_joint_pointcloud_pose_test.yaml \
  --stage export
```

## 11. 方法边界

- 当前是单场景 proof-of-concept，不是跨场景泛化结论；
- 它是学习残差前端加解析几何后端，不是迭代 bundle adjustment；
- refined pose 不会反向再次更新 pointmap；
- V4 通过空间隔离和原始输出回退限制 pointmap 负迁移，但实际效果仍由新实验决定；
- reference mask 来自数据集实例标注，不是全自动实例发现；
- 关闭 adapter 时严格恢复原始 StreamVGGT；ray fit 失败时回退 learned center。
