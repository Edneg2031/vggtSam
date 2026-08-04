# SAM3.1 辅助 StreamVGGT 的当前方法（V7.4）

> 文档状态：当前实现，更新于 2026-08-04。
> 本文是当前方法的唯一总说明；V6、V7.1、V7.2、V7.3 文档仅作为历史实验和复现手册保留。

## 1. 一句话结论

当前方法不是把一个 SAM persistent vector 直接加到 StreamVGGT token 上，也不是让 SAM3.1
直接预测相机位姿。SAM3.1 负责发现和持续追踪物体，并用 mask 内的局部语义 descriptor
决定跨帧局部对应权重；真正被传输、送入位姿修正头的 Value 始终是 StreamVGGT 的局部几何。
这些实例证据在多物体间加权汇聚后，与冻结的 camera-only 基线特征融合，预测一个有界的
SE(3) residual。

因此，SAM3.1 对当前 StreamVGGT 的直接作用是**帮助跨帧几何对应和相机位姿修正**；它不直接
更新 depth 或 dense pointmap。点云若有改善，只能来自更稳定的相机位姿带来的间接对齐改善。

## 2. 当前数据流

```text
图像序列
  ├─ StreamVGGT（冻结）
  │    ├─ camera hidden / 原始 world-to-camera pose
  │    ├─ dense world pointmap + confidence
  │    └─ mask 内局部几何 token
  │
  └─ SAM3.1 multiplex（冻结，forward-only，具体概念提示）
       ├─ 动态发现 persistent object ID
       ├─ 每帧实例 mask 与 track confidence
       └─ mask 内 detector_fpn2 局部 descriptor

同一实例的当前观测
  → 与该实例“上一条可靠历史观测”匹配
  → SAM identity affinity + StreamVGGT geometry affinity
  → transport 历史 StreamVGGT geometry Value
  → 单实例证据
  → reliability-weighted 多实例汇聚
  → 与 frozen L0 camera feature 融合
  → zero-initialized SE(3) residual
  → refined camera pose
```

这里不存在“第 90 帧必须出现某个物体”的要求。第 90 帧只固定相机坐标系 gauge，物体可以在
后续任意帧首次出现。

## 3. 动态实例发现与生命周期

### 3.1 不再依赖固定 GT instance ID

V7.4 使用 SAM3.1 multiplex tracker 对整个序列做严格的 forward-only 处理：

- SAM3.1 不是 class-agnostic proposal model，当前场景分别使用 `bed` 和 `wardrobe`
  两个具体概念提示，再按出生时间合并各自的 persistent track；
- 当前配置提供 8 个逻辑容量 slot `[0,1,...,7]`，它们不是 ScanNet GT ID；
- 新的 SAM persistent ID 第一次出现时分配 slot；
- slot 按首次出现顺序分配，分配后不复用；
- 多个提示若在新 track 的出生帧产生高度重叠 mask，会以因果方式去重；
- 只向前传播，不用后续帧反向修正过去结果；
- GT mask 和 GT instance ID 不参与实例发现、slot 分配或 correspondence。

GT camera pose 仍然用于有监督的 pose loss 和最终评估，这是训练信号，不是推理输入。
概念提示属于推理配置，因此更换场景时需要给出可能作为静态锚点的名词列表；这仍然不需要
指定物体 ID、参考帧 mask 或物体第一次出现的帧号。

### 3.2 SAM birth 与 geometry birth

一个实例会经历两个不同事件：

1. **SAM birth**：SAM3.1 第一次输出该 persistent ID 和有效 mask；
2. **geometry birth**：该 mask 内第一次具有足够的 StreamVGGT 三维点，能够初始化实例几何
   memory。

geometry birth 帧只写入 memory，不允许修正该帧位姿。只有实例在之后再次被可靠观测时，才
同时拥有“历史 Key/Value”和“当前 Query”，从而成为 mature track 并参与 pose correction。
这避免模型把单帧实例特征当作相机位姿捷径。

### 3.3 可靠性与 memory 写入

当前帧必须至少满足以下条件，才可能写入或使用实例 memory：

- SAM mask 确实存在；
- track confidence 达到阈值；
- mask 内存在有效的 StreamVGGT 局部几何；
- identity 状态没有被几何一致性判为硬 mismatch。

matcher 使用 `causal_last_observation`：第 `t` 帧只能读取严格早于 `t` 的最近一次可靠观测，
然后才决定是否将第 `t` 帧写回 memory。它不再像 V7.3 那样始终与固定物体参考帧匹配。

## 4. 两类局部 token

### 4.1 SAM3.1 局部 identity descriptor

SAM3.1 的 `detector_fpn2` 特征保留为空间网格，而不是全局池化成一个向量。对每个实例 mask：

- 将 mask 对齐到 `72 × 72` 特征网格；
- 在 mask 内用 deterministic farthest-UV sampling 取最多 32 个局部 token；
- cache 保存 descriptor、归一化 UV 和 valid mask；
- V7.4 正式消融使用 `K=8`，其余 token 留作后续尺度实验。

这些 descriptor 的职责是表达局部语义/外观身份，例如当前几何点更可能对应历史物体的哪个
局部区域。

### 4.2 StreamVGGT 局部 geometry token

在同一个实例 mask 内，从 StreamVGGT 的 pointmap/depth/confidence 构造局部几何 token。当前
每个 token 是 7 维：

```text
[normalized world xyz, normalized uv, log depth, normalized confidence]
```

几何 token 具有真实空间含义，是 correspondence 中的 Query、Key 和最终被 transport 的
Value。SAM token 不替代几何 Value。

### 4.3 UV 对齐

SAM 网格与 StreamVGGT 几何采样点不完全相同。实现先依据归一化 UV 距离，把附近 SAM
descriptor 软插值到每个 StreamVGGT 几何点：

```math
s_i = \sum_m \operatorname{softmax}_m\left(-\frac{\|u_i-u_m^{sam}\|^2}{\tau_{uv}}\right) s_m^{sam}
```

UV 在这里只负责跨特征网格对齐，不被编码成可学习的位姿捷径。

## 5. SAM 如何真正改变几何对应

对当前帧的几何点 `i` 与该实例上一条可靠历史观测中的几何点 `j`，主分支
`sam_geometry_transport` 计算两类相似度：

```math
\ell_{ij}^{geo} = \frac{\cos(W_q^g g_i, W_k^g g_j^{hist})}{\tau}
```

```math
\ell_{ij}^{sam} = \frac{\cos(W_q^s s_i, W_k^s s_j^{hist})}{\tau}
```

二者相加后做 masked softmax：

```math
p_{ij} = \operatorname{softmax}_j(\ell_{ij}^{geo}+\ell_{ij}^{sam})
```

相加 logits 等价于一个 product-of-experts：只有几何和 SAM identity 都支持的对应才获得较高
权重。随后 transport 的仍是历史 StreamVGGT geometry：

```math
\hat g_i^{hist} = \sum_j p_{ij} g_j^{hist}
```

单点 residual encoder 的输入为：

```text
[current geometry,
 transported historical geometry,
 current - historical,
 current * historical]
```

点级证据按 StreamVGGT point confidence 汇聚为一个实例证据。这样避免了“只有一个 Key 的
cross-attention 经过 softmax 后恒为 1”的退化问题：当前 attention 的 Key/Value 长度是多个
局部点，而不是单个 persistent vector。

## 6. 从多实例证据到相机位姿

同一帧可以有多个 mature instance。每个实例的证据按 track confidence 和 identity
reliability 加权汇聚：

```math
e_t = \frac{\sum_k r_{t,k} e_{t,k}}{\sum_k r_{t,k}}
```

然后将实例证据 `e_t` 与冻结 camera-only L0 特征 `c_t` 融合：

```text
f_t = MLP([c_t, e_t, c_t * e_t, c_t - e_t])
ξ_t = SE3Head(f_t)
T_t_refined = Exp(ξ_t) · T_t_L0
```

SE(3) head 最后一层零初始化，因此训练开始时严格复现 frozen L0。旋转和平移 residual 都经过
有界映射；第 90 帧的 residual 被强制为零，以保持坐标 gauge 不变。

当某帧没有 mature instance、没有历史 geometry Key/Value，或实例 gate 全部失败时，
`active_frames=False`，输出逐元素精确回退 frozen L0，而不是生成一次无依据的修正。

## 7. 训练目标到底优化什么

V7.4 当前只训练相机位姿 residual，不训练 SAM3.1、StreamVGGT、depth 或 dense pointmap。
损失是：

```math
L_{pose} = \operatorname{MSE}(R_{pred}, R_{gt})
         + 10\,\operatorname{SmoothL1}(C_{pred}, C_{gt};\,\beta=0.01)
```

其中 `R` 是旋转矩阵，`C=-R^Tt` 是 camera center。CSV 中同时给出更直观的：

- `test_rotation_deg`：平均旋转角误差；
- `test_translation`：camera center 的原生坐标误差；
- `test_loss`：上述训练目标；
- `gain_vs_frozen_l0_percent`：相对冻结 camera-only L0 的提升比例。

所以“loss 降到 0”只说明训练帧上的 pose residual 具有拟合能力，不能证明 SAM token 提供了
泛化增益。SAM 是否真正有用必须由下一节的 matched controls 和时间外推共同判断。

## 8. 当前 V7.4 验证协议

序列为 `90:15:525`。frame 90 只作为 gauge；camera-only L0 使用早期
`105–255` 训练并被冻结。correspondence residual 使用严格按时间划分的三组实验：

| Fold | residual 训练帧 | 未见测试帧 |
|---|---|---|
| short | 270–345，步长 15 | 360–405，步长 15 |
| medium | 270–405，步长 15 | 420–465，步长 15 |
| long | 270–465，步长 15 | 480–525，步长 15 |

这是**同一场景内的时间前缀泛化**，不能表述为跨场景泛化。

主候选必须同时面对下列控制：

| 分支/扰动 | 回答的问题 |
|---|---|
| `raw_streamvggt` | 原始 StreamVGGT 水平 |
| `frozen_l0` | 不使用实例的 camera-only 学习基线 |
| `uniform_transport` | 提升是否只来自更多参数和局部 Value |
| `geometry_transport` | 不用 SAM descriptor 时，纯几何对应能做到多少 |
| `sam_transport` | 只用 SAM affinity 是否有效 |
| `sam_geometry_transport` | SAM + geometry 是否互补 |
| `sam_geometry_train_sam_off` | 相同结构在训练时关闭 SAM 后的 matched control |
| `sam_off` / `uniform_sam` | 推理时移除或抹平 SAM 是否损害结果 |
| `wrong_sam_identity` | 错误实例身份是否损害结果 |
| `shuffle_sam_time` | 打乱时间对应是否损害结果 |
| `wrong_local_geometry` | 错误几何 Value 是否损害结果 |

只有 `sam_geometry_transport` 在相同 active-frame support 下优于纯几何和 trained-SAM-off
控制，并且移除、抹平、错配或打乱 SAM 后都稳定变差，才能认为提升来自 SAM correspondence，
而不是额外参数、门控覆盖率或模型单纯过拟合 pose。

## 9. SAM 对点云的作用边界

当前 V7.4：

- **不会**修改 StreamVGGT 每帧预测的 depth；
- **不会**修改 dense pointmap token 或 point coordinate；
- **没有**使用 pointmap loss、depth loss、Chamfer distance 或 GT object pointcloud loss；
- **不会**直接生成“优化后的物体形状”。

若用 refined camera pose 将各帧局部点云变换到共同坐标系，位姿漂移减小可能使拼接地图和物体
点云更集中。但这属于 pose 改善的间接结果：单帧深度错误、物体表面噪声和 mask 错误仍然存在。
因此汇报时应说“SAM 辅助跨帧几何对应并优化相机位姿，从而可能改善点云对齐”，不能说
“SAM 已直接优化 pointmap”。

## 10. 与旧版本的关键区别

| 版本 | 主要设计 | 当前定位 |
|---|---|---|
| V4/V5 | global instance token、camera/pointmap/ray solver 多分支 | 历史方法，不代表当前实现 |
| V6 | SAM3.1 mask 与 pooled instance vector 的 camera residual/容量实验 | 证明可拟合与初步提升，但难以排除容量捷径 |
| V7/V7.1 | 从 monolithic fusion 转为冻结 L0 后测试实例增量 | 建立公平 camera-capacity 和 gate control |
| V7.2 | 保留 mask 内多个 SAM local token | 解决单向量 attention 退化，但 Value 来源仍需解耦 |
| V7.3 | SAM 只做 correspondence weight，geometry 做 Value | 方法方向确立，但依赖固定物体参考帧 |
| **V7.4** | 动态实例出生 + causal last-observation memory + temporal folds | **当前实现与当前验证协议** |

## 11. 运行与输出

服务器从仓库根目录运行：

```bash
zsh streaming_couping/commands_v74_temporal_scaling.txt
```

如需彻底重建 backbone cache 和所有训练结果：

```bash
V74_FRESH=1 zsh streaming_couping/commands_v74_temporal_scaling.txt
```

主要输出：

```text
outputs/streaming_couping_v74_temporal_scaling/v74_temporal_scaling.csv
outputs/streaming_couping_v74_temporal_scaling/v74_dynamic_instance_diagnostics.csv
outputs/streaming_couping_v74_temporal_scaling/v74_temporal_scaling_metadata.json
outputs/streaming_couping_v74_temporal_scaling/frozen_l0.pt
```

`v74_temporal_scaling.csv` 是判断过拟合能力、时间外推和 SAM 因果贡献的主表；
`v74_dynamic_instance_diagnostics.csv` 用于检查每帧发现多少 track、哪些 slot 出生、多少实例已
mature，以及为何某帧回退 L0。

## 12. 代码对应关系

| 功能 | 文件 |
|---|---|
| SAM3.1 forward-only 动态追踪 | `src/backbones/sam3_video.py` |
| online slot 分配与 cache | `src/learned_pose/cache.py` |
| geometry birth、identity 与因果实例观测 | `src/learned_pose/observations.py` |
| SAM/geometry affinity、transport、SE(3) residual | `src/learned_pose/v73_correspondence_fusion.py` |
| V7.4 时间 fold、训练、控制和 CSV | `scripts/run_v74_temporal_scaling.py` |
| 数据与设备配置 | `configs/v74_temporal_data.yaml` |
| 一键服务器命令 | `commands_v74_temporal_scaling.txt` |

上述相对路径均以 `streaming_couping/` 为根。

## 13. PPT 可直接使用的表述

> 我们不把 SAM3.1 压成单个全局向量后直接注入相机 token，而是保留 mask 内多个局部语义
> descriptor。SAM3.1 在视频中动态发现并追踪实例，语义 descriptor 与 StreamVGGT 几何特征
> 共同计算当前观测到最近可靠历史观测的局部对应；attention 只决定对应权重，被传输的 Value
> 始终是 StreamVGGT 几何。多实例证据经过可靠性加权后预测冻结 camera-only 基线之上的有界
> SE(3) residual。没有成熟实例时严格回退基线。当前实验优化和评估的是相机 pose，点云仅可能
> 通过更准确的跨帧位姿获得间接对齐改善。
