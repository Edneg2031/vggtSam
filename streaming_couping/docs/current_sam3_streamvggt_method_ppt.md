# SAM3 + StreamVGGT 实例引导相机位姿优化方法

> 本文描述当前代码实际实现的方法，可直接拆分为 PPT。核心思想是：**SAM3 提供跨帧实例观测，StreamVGGT 提供相机特征与三维几何验证，persistent instance memory 保存可信实例历史，V6 camera–instance fusion 修正原始相机位姿。**

## 1. 方法目标

StreamVGGT 能够流式预测相机位姿、深度和三维点，但长序列中可能产生旋转抖动、平移漂移和点云重影。静态物体在多帧中应保持一致，因此可以作为额外的几何锚点。

本方法希望利用 SAM3 持续跟踪的物体回答两个问题：

1. 当前相机运动是否与历史静态物体的观测一致？
2. 如果不一致，如何利用实例信息修正 StreamVGGT 的相机位姿？

## 2. 整体流程

```text
输入视频
   │
   ├── StreamVGGT
   │     ├── 原始相机位姿 Traw
   │     ├── Camera hidden token
   │     └── Dense world pointmap
   │
   └── SAM3
         ├── 参考帧文本检测
         ├── 跨帧 mask 跟踪
         └── mask 内 FPN2 外观特征
                │
                ▼
       StreamVGGT 三维几何验证
                │
                ▼
       Persistent instance token
                │
                ▼
       Camera–instance cross attention
                │
                ▼
       位姿修正 Trefined
```

当前系统中，SAM3 先完成实例检测与传播，StreamVGGT 再检查跟踪结果是否满足三维一致性。因此准确表述是：

> **SAM3 提出跨视角实例匹配，StreamVGGT 对匹配进行三维身份验证。**

StreamVGGT 当前不会将三维投影重新输入 SAM3，也不会在几何不一致时要求 SAM3 内部重新分割。

## 3. SAM3 实例观测

### 3.1 参考帧初始化

在参考帧中使用文本提示检测目标，例如会议室实验中的：

```text
chair
monitor
```

每个文本提示选择一个有效 mask，并建立一个独立的 persistent instance slot。之后 SAM3 视频跟踪器从参考帧开始向后传播对应 mask。

### 3.2 实例外观特征

当前使用的实例特征不是 SAM3 最终的 mask token，而是 SAM3 detector 的 **FPN2 空间特征**。对每个实例 mask 内的特征计算均值和标准差：

\[
a_{t,k}=
\left[
\operatorname{Mean}(F_t\mid M_{t,k}),
\operatorname{Std}(F_t\mid M_{t,k})
\right]
\]

其中：

- \(F_t\) 是第 \(t\) 帧的 SAM3 FPN2 特征；
- \(M_{t,k}\) 是实例 \(k\) 在第 \(t\) 帧的 mask；
- \(a_{t,k}\) 是 mask 池化后的实例外观描述。

外观特征能够支持跨视角识别，但相似外观的不同椅子、显示器或柜子仍可能被错误关联，因此还需要三维几何验证。

## 4. StreamVGGT 三维几何验证

### 4.1 从 mask 获得实例点集

StreamVGGT 官方 point head 输出每个像素对应的世界坐标：

\[
X_t(u)\in\mathbb{R}^3
\]

根据 SAM3 mask 和点置信度，提取当前实例的三维点集：

\[
P_{t,k}=\left\{
X_t(u)\mid u\in M_{t,k},\;conf(u)>\tau
\right\}
\]

参考帧的点集被保存为该实例的初始三维地图。

### 4.2 跨帧几何配准

后续帧的实例点集与历史实例地图进行有界、平移型 ICP 配准，并计算：

- 三维对应点数量；
- registration fitness；
- registration RMSE；
- 三维形状相似度；
- 多实例运动提议之间的一致性；
- tracking、geometry 和 static confidence。

这些指标用于判断 SAM3 当前跟踪结果是否仍然属于参考实例。

### 4.3 身份状态

| 状态 | 几何含义 | 后续处理 |
|---|---|---|
| `MATCH` | 当前点集与历史实例地图能够可靠配准 | 允许参与位姿融合并更新实例记忆 |
| `UNKNOWN` | 点数或几何支持不足，无法确认也无法否定 | 降低可靠性使用，不允许更新记忆 |
| `MISMATCH` | mask 有足够三维点，但与历史结构几乎没有对应 | 排除该实例观测，防止污染记忆 |

因此，StreamVGGT 对 SAM3 跨视角匹配的帮助不是重新预测 mask，而是：

> **将二维外观跟踪提升到三维空间，对实例身份进行几何验证。**

## 5. Persistent instance token

### 5.1 为什么需要持久实例

单帧 mask 只能表示当前观测，不能表达物体的历史外观和几何。系统为每个实例保存一份因果记忆，使其成为跨帧稳定的场景锚点。

### 5.2 Token 输入

每个实例 token 组合以下信息：

```text
当前外观
历史外观记忆
当前外观 - 历史外观

当前几何
历史几何记忆
当前几何 - 历史几何

tracking / geometry / static confidence
OBSERVED / MATCH / UNKNOWN
memory age
```

这些特征经过 MLP 编码为 persistent instance token：

\[
p_{t,k}=\operatorname{MLP}
\left[
a_{t,k},\bar a_{t-1,k},a_{t,k}-\bar a_{t-1,k},
g_{t,k},\bar g_{t-1,k},g_{t,k}-\bar g_{t-1,k},
q_{t,k}
\right]
\]

### 5.3 记忆更新

只有经过几何验证的 `MATCH` 观测才能更新记忆：

\[
\bar a_{t,k}
=
\mu\bar a_{t-1,k}+(1-\mu)a_{t,k}
\]

当前实现使用 \(\mu=0.9\)。几何记忆采用相同的指数滑动更新。

- `MATCH`：可以读取并更新记忆；
- `UNKNOWN`：可以低可靠性地与记忆比较，但不能写入；
- `MISMATCH`：不能参与当前实例融合，也不能更新记忆。

这能够降低错误 mask 对长期实例状态的污染。

## 6. Camera–instance fusion

### 6.1 Camera token

当前使用的是 StreamVGGT aggregator 最后一层中、送入官方 CameraHead 之前的 camera hidden token：

\[
c_t=\operatorname{Project}(h_t^{camera})
\]

它包含 StreamVGGT 对当前图像和历史视频上下文的相机运动表征。

### 6.2 Cross attention

以 camera token 为 Query，以当前所有可用 persistent instance token 为 Key 和 Value：

\[
\hat p_t=
\operatorname{CrossAttention}
\left(c_t,\{p_{t,k}\}\right)
\]

可靠性较高的实例在 attention 中具有更高权重；`MISMATCH` 实例不会参与 attention。

### 6.3 特征融合

当前实现不是简单的加法残差，而是显式构造 camera 与 instance 的关系：

\[
z_t=
\operatorname{MLP}
\left[
c_t,\hat p_t,
c_t\odot\hat p_t,
c_t-\hat p_t
\right]
\]

其中：

- \(c_t\) 保留原始 camera 表征；
- \(\hat p_t\) 表示实例提供的运动约束；
- \(c_t\odot\hat p_t\) 表示两种特征的一致部分；
- \(c_t-\hat p_t\) 表示 camera 与实例之间的差异。

融合后的新特征 \(z_t\) 被送入受限的位姿修正 head。

## 7. 位姿修正方案

### 7.1 V6 Fusion

融合特征直接预测受限的 SE(3) 修正：

\[
T'_t=\exp(\delta\xi_t)T_t^{raw}
\]

该分支同时修正旋转和平移。

### 7.2 候选 A

候选 A 对旋转和相机中心进行职责解耦：

```text
rotation ← camera token 的 rotation head
center   ← instance token 的 3DoF local-center head
```

旋转主要由全局 camera token 决定，实例负责提供局部平移和相机中心约束。

### 7.3 候选 B

候选 B 仍由 camera token 负责最终旋转。实例分支训练时使用完整 SE(3) head，并加入权重为 `0.1` 的辅助旋转监督；推理时丢弃实例分支预测的旋转，只提取其相机中心：

```text
rotation ← camera token
center   ← instance SE(3) head
           训练时加入弱旋转辅助监督
           推理时只使用 center
```

候选 B 用于测试弱旋转监督是否能够让实例 representation 学到更稳定的三维运动信息。

### 7.4 安全回退

参考帧保持原始 StreamVGGT 位姿不变。没有可用 persistent instance 支持的帧不应用实例修正，输出回退到原始 StreamVGGT 位姿。

## 8. 训练与部署

### 8.1 训练阶段

训练时：

- 冻结 SAM3；
- 冻结 StreamVGGT；
- 不训练 depth head 和 point head；
- 只训练 persistent instance encoder、cross attention、feature merger 和位姿修正 head；
- 使用旧场景的 GT pose 监督位姿修正。

因此，训练的目标不是让模型重新学习完整相机位姿，而是学习：

> **在 StreamVGGT 已有位姿基础上，如何根据 camera–instance 不一致预测一个有界修正。**

### 8.2 新场景部署

会议室实验中：

- 不使用会议室 GT pose、GT mask 或 GT instance ID；
- 不重新训练模型；
- 使用旧场景训练完成的固定 V6 checkpoint；
- SAM3 通过 `chair` 和 `monitor` 自动初始化实例；
- 在 300 帧长序列上测试跨场景泛化。

## 9. 物体点云对比

对于 chair 和 monitor，固定使用同一组 SAM3 mask，分别导出：

```text
streamvggt_raw
v6_fusion
v6_candidate_a
v6_candidate_b
horizonstream
streamvggt_pointhead
```

前五组主要比较 depth 与不同 pose 组合后的多视图重合；`streamvggt_pointhead` 是直接预测的世界点，用于隔离 point head 本身的能力。

由于 HorizonStream 和 StreamVGGT 的坐标 gauge、尺度各自独立，不能直接把原生点云坐标差解释成误差。

## 10. 当前方法边界

当前已经实现：

- SAM3 开放词汇参考帧初始化；
- SAM3 跨帧 mask 传播；
- StreamVGGT 三维实例身份验证；
- persistent instance memory；
- camera–instance cross attention；
- V6 位姿修正和无实例回退；
- 实例 mask、轨迹和物体点云导出。

当前尚未实现：

- 将三维投影作为 box、point 或 mask prompt 重新输入 SAM3；
- 在几何 `MISMATCH` 时自动触发 SAM3 重检测；
- 利用多候选 mask 的三维得分重新选择 SAM3 实例；
- 新场景上的 GT 位姿精度评价。

因此汇报时不应说“StreamVGGT 在 SAM3 内部完成了跨视角匹配”，而应说：

> **SAM3 负责生成跨视角实例候选，StreamVGGT 通过三维几何验证候选身份，可信实例再作为持久场景锚点参与相机位姿修正。**

## 11. PPT 一页总结

> 本方法首先利用 SAM3 在参考帧检测并跨帧跟踪物体，然后利用 StreamVGGT 的世界点图，将每帧实例 mask 提升为三维点集，并与实例的历史三维地图进行配准。三维一致性用于区分可信匹配、几何不确定和明显错配，只有可信观测才能更新 persistent instance memory。随后，以 StreamVGGT 最后一层 camera token 为查询，与包含实例外观、几何和历史信息的 persistent instance token 进行 cross attention，再通过融合特征预测受限的相机位姿修正。实例不是替代相机模型，而是作为跨帧稳定的场景锚点，帮助判断和修正 StreamVGGT 的相机运动。

