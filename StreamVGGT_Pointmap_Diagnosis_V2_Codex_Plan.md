# StreamVGGT + SAM3.1：Pointmap 下一阶段诊断实验方案

## 0. 当前结论与本轮目标

当前正式 V0 保持不变：

```text
Pose:
first-frame anchor + native QK Top-4
→ frozen StreamVGGT camera head
→ QK pose

Geometry:
full-history StreamVGGT point head
→ raw world pointmap

Semantic:
SAM3.1 persistent mask / slot / track ID
→ semantic lifting
```

已有审计结论：

- QK context 能改善 camera pose；
- 同一 QK context 下，depth head 可以改善，但 direct point head 反而下降；
- full-history raw pointmap 目前仍是最稳定 geometry baseline；
- shared-SE(3)、joint BA、object ICP、surfel/raw-to-raw refinement、SAM memory retrieval 等路线已经做过，不要原样重复；
- SAM persistent ID 已经证明能找到更相干的跨帧实例集合，但尚未证明能给出正确的 metric point update direction；
- “2D correspondence + multi-view rays → independent 3D anchors”还没有真正做过。

本轮实验的目标不是立刻改善 pointmap，而是回答：

> **raw pointmap 为什么不准，以及能否产生一种不依赖 raw pointmap 自一致性的独立几何证据。**

本轮禁止训练模型，禁止修改正式 V0 pointmap，禁止实现复杂 optimizer。

---

# 1. 本轮实验顺序

严格按下面顺序执行：

```text
D0. Depth × Pose 六格 Oracle
        ↓
判断 depth / pose / unprojection 哪个是主要瓶颈

D1. Raw Pointmap Error Decomposition
        ↓
判断错误集中在 boundary、interior、background 还是 confidence 失效

T0. SAM-Indexed Multi-View Triangulation Probe
        ↓
第一次产生独立于 raw pointmap 的 sparse 3D anchor

只有 T0 成立
        ↓
后续才讨论 T1 pointmap refinement
```

本轮只做到 T0。

---

# 2. D0：Depth × Pose 六格 Oracle

## 2.1 目的

当前存在一个未解决的问题：

```text
direct raw pointmap
明显好于
depth + pose backprojection map
```

历史 joint BA 的 `0.1526 → 0.5087` 不能直接用于判断 depth head 是否差，因为其中混杂了：

- direct XYZ → z-depth → ray XYZ；
- pose 改动；
- fixed reference K；
- depth blend；
- coordinate/gauge 变化。

因此需要做严格 factorization。

---

## 2.2 六个组合

统一：

- 同一 scene；
- 同一 frame range；
- 同一 valid support；
- 同一 intrinsics convention；
- 同一 gauge；
- 同一 alignment；
- 同一 evaluator。

运行：

```text
A. raw depth + raw pose
B. raw depth + QK pose
C. raw depth + GT pose

D. GT depth + raw pose
E. GT depth + QK pose
F. GT depth + GT pose
```

另外保留：

```text
G. direct full-history raw pointmap
```

作为 point-head baseline。

---

## 2.3 必须检查的实现细节

在运行前先记录：

```text
depth representation:
    camera-space z-depth / ray depth / inverse depth ?

pose convention:
    world_to_camera / camera_to_world ?

intrinsics:
    每帧 K / reference K / predicted K / GT K ?

unprojection:
    X_c = K^-1 [u,v,1]^T * depth
    X_w = T_c2w X_c
```

必须做一个 sanity check：

```text
GT depth + GT pose + GT K
```

如果这个组合都不能恢复低误差 world points，则先停止所有方法实验，修正坐标 / K / evaluator。

---

## 2.4 输出

保存：

```text
outputs/.../d0_depth_pose_oracle/
    summary.json
    per_frame.csv
    config.json
```

表格至少包含：

| Depth | Pose | K | Paired RMSE | Symmetric / Chamfer | Valid Ratio |
|---|---|---|---:|---:|---:|

---

## 2.5 D0 决策

### Case A

```text
raw depth + GT pose
仍然明显差
```

说明：

> pointmap 问题主要不是 pose，raw depth / depth parameterization 本身存在明显误差。

后续不要把“QK pose + depth”作为主要 pointmap 修正来源。

### Case B

```text
raw depth + GT pose
明显变好
```

说明：

> depth 有潜力，pose / coordinate coupling 是主要问题之一。

后续 triangulation / depth consistency 可以继续使用 raw depth。

### Case C

```text
GT depth + QK pose
已经接近 GT depth + GT pose
```

说明：

> QK pose 已经足够支持几何 reconstruction，主要瓶颈在 dense geometry，而不是 camera。

### Case D

```text
GT depth + GT pose
仍然差
```

说明：

> implementation / K / gauge / evaluator 有错误。

立即停止 D1/T0，先修 pipeline。

---

# 3. D1：Raw Pointmap Error Decomposition

## 3.1 目的

现在还没有证据说明 raw pointmap 的误差集中在：

- object boundary；
- object interior；
- background；
- low-confidence region；
- high-confidence-but-wrong region。

SAM 是否能帮助 geometry，必须先看错误到底发生在哪里。

---

## 3.2 使用 SAM3.1 做区域划分

对每帧 persistent mask：

```text
mask interior:
    对 mask erosion 后的区域

mask boundary:
    原 mask - eroded mask
    或固定宽度 boundary ring

background:
    不属于任何有效 SAM instance 的像素

unlabeled:
    根据当前 pipeline 定义单独统计
```

建议至少测试两种 boundary width：

```text
3 px
5 px
```

不要因为某一个 width 得出结论。

---

## 3.3 计算 raw pointmap GT error

每个有效像素：

```text
e_point = ||P_raw_world - P_gt_world||
```

对以下区域分别统计：

```text
instance interior
instance boundary
background
unlabeled
```

输出：

- count；
- mean；
- median；
- P90；
- RMSE。

---

## 3.4 按 persistent instance 统计

每个 slot 输出：

```text
slot
prompt
track_id
birth_frame
visible_frame_count
point_count

raw_error_mean
raw_error_median
raw_error_p90

interior_error
boundary_error
boundary / interior ratio
```

目的：

> 看 geometry error 是否和 instance 类型、观测次数、边界复杂度有关。

---

## 3.5 Confidence calibration

StreamVGGT point head 有独立 confidence。

做：

```text
confidence quantile:
0–20%
20–40%
40–60%
60–80%
80–100%
```

每个 bin 输出：

```text
count
mean GT error
median GT error
P90
```

并额外统计：

```text
high-confidence wrong points
```

例如：

```text
confidence top 20%
但 GT error > overall P90
```

所占比例。

如果方便，再输出 risk-coverage curve。

---

## 3.6 D1 决策

### Case A：Boundary 明显更差

若：

```text
boundary error >> interior error
```

且在多个 instance / frame 稳定成立：

说明 SAM 有一种真实的新 geometry information：

> **object boundary / ownership / depth discontinuity prior**

后续可考虑 boundary-aware geometry refinement。

### Case B：Interior 也一样差

如果：

```text
boundary ≈ interior
```

说明 pointmap 的主要错误不是跨物体污染。

此时不能期待单纯 SAM mask / erosion / boundary loss 大幅改善 XYZ。

### Case C：Confidence 能有效识别错误

如果：

```text
higher confidence → clearly lower GT error
```

后续可以只对 low-confidence raw geometry 引入新 anchor。

### Case D：High-confidence points 也大量错误

说明 StreamVGGT confidence 不能直接作为 geometry correctness gate。

---

# 4. T0：SAM-Indexed Multi-View Triangulation Probe

## 4.1 核心动机

以前失败路线主要是：

```text
raw pointmap A
      ↕
raw pointmap B
      ↓
根据内部一致性移动 raw points
```

问题：

> raw-to-raw consistency 不是独立真值。

T0 第一次尝试构造：

```text
2D correspondence
+
multi-view camera rays
        ↓
independent 3D anchor
```

目标不是修改 pointmap，而是先验证：

> **triangulated 3D 是否比 raw pointmap 更接近 GT。**

---

# 5. SAM 在 T0 中只负责“应该和谁比较”

不要让 SAM 直接预测 XYZ。

SAM3.1 的作用定义为：

```text
persistent slot / track ID
        ↓
约束 correspondence 必须来自同一 physical instance
```

例如：

```text
current chair
只能和
historical chair
做 correspondence
```

禁止：

```text
chair ↔ table
chair ↔ cabinet
```

因此：

```text
SAM:
解决 cross-frame identity / ownership

2D correspondence:
解决物体内部具体 pixel ↔ pixel

camera rays:
提供 metric geometric constraint

triangulation:
产生 independent sparse 3D
```

---

# 6. T0 的 history selection

不要简单：

```text
same persistent ID
→ 所有历史帧全部使用
```

已有历史证据表明：

> 同一个 current frame 中，不同 history edge 的 conditioning 差异可能极大。

因此 history selection 分两级。

---

## 6.1 Stage 1：SAM co-visibility filter

对 current instance `i`：

```text
candidate_history_i =
    所有历史中
    persistent slot i 可见的 frame
```

记录：

```text
mask area
SAM score
visible history length
temporal distance
```

---

## 6.2 Stage 2：geometry suitability filter

从 candidate history 中选择适合 triangulation 的 views。

优先依据：

```text
camera baseline
rotation difference
ray angle
epipolar / essential condition
track reprojection consistency
```

第一版不要使用复杂 learned selector。

最重要的是过滤：

```text
baseline 太小
接近平行 ray
condition number 过大
明显 degenerate edge
```

---

# 7. 2D Correspondence Source

不要默认 TrackHead 一定可用。

历史 TrackHead-BA 出现过：

```text
visibility/confidence gate = 0 / 3072
```

所以本轮至少审计两种 correspondence source。

---

## T0-A：已有 V9 visible-surface / epipolar correspondence

优先复用历史上已经证明 relative pose 有上界的 correspondence pipeline。

要求：

- 不使用 raw 3D pointmap 形成 landmark；
- 只使用 2D correspondence；
- 输出 match confidence / condition。

---

## T0-B：StreamVGGT TrackHead

如果当前 TrackHead 可以修正调用方式并正常输出：

```text
coords
visibility
confidence
```

则单独做 TrackHead triangulation probe。

注意：

> 不要复用旧 TrackHead-BA 中“predicted depth 作为 3D landmark”的逻辑。

T0 必须：

```text
2D track
→ ray
→ triangulation
```

而不是：

```text
2D track + raw depth
→ 3D landmark
```

如果 TrackHead 仍然无法产生有效 visibility/confidence：

```text
标记 TRACKHEAD_UNUSABLE
```

不要为了 T0 强行调大工程量。

---

# 8. Triangulation

对于同一 instance 内的一个 track：

```text
u_1, u_2, ..., u_n
```

对应相机：

```text
K_1, T_1
K_2, T_2
...
```

构造 rays：

```text
r_k(u_k)
```

通过 multi-view triangulation 得到：

```text
X_tri
```

建议：

- 先 linear triangulation；
- 再少量 reprojection refinement；
- 使用 robust loss；
- 至少 2 views；
- 优先 3+ views；
- 不更新任何 network / pose / map。

---

# 9. T0 Validity Gate

每个 triangulated anchor 至少记录：

```text
num_views
ray_angle / minimum baseline
condition number
mean reprojection error
max reprojection error
SAM slot consistency
track confidence
visibility
```

建议第一版 gate：

```text
num_views >= 2 or 3
condition < threshold
reprojection_error < threshold
all observations same persistent slot
```

阈值不要拍脑袋固定。

先输出分布，再选合理 threshold。

---

# 10. T0 核心评价

对每个 triangulated anchor：

```text
e_tri = ||X_tri - X_gt||
```

同时找到对应 raw pointmap observation：

```text
e_raw = ||X_raw - X_gt||
```

比较：

```text
delta = e_raw - e_tri
```

其中：

```text
delta > 0
```

表示 independent triangulation 比 raw pointmap 更好。

---

## 10.1 必报指标

整体：

```text
valid triangulated anchor count
valid rate
triangulation RMSE / median / P90
raw matched-point RMSE / median / P90
improved anchor ratio
mean delta
median delta
```

---

## 10.2 分组分析

按以下变量分组：

```text
2 views / 3 views / 4+ views
condition number
ray angle
camera baseline
track confidence
SAM interior / boundary
persistent instance
visible history count
raw point confidence
```

特别关心：

> 是否存在一个明确的 high-quality subset，使 `X_tri` 稳定优于 `X_raw`。

不要求所有 triangulated points 都优于 raw。

---

# 11. T0 Control

至少做：

```text
A. correct persistent ID
B. shuffled persistent ID
```

如果成本允许再加：

```text
C. foreground-only，不区分 instance
D. shifted mask
```

目标不是直接看 pointmap improvement，而是看：

```text
valid triangulation rate
reprojection error
GT anchor error
```

理想结果：

```text
correct ID
→ 更多有效 triangulation
→ 更低 reprojection
→ 更低 GT 3D error
```

如果 shuffled ID 也一样好：

说明 SAM identity 并没有真正帮助 independent geometry。

---

# 12. 本轮禁止事项

不要：

```text
❌ 修改 raw world pointmap
❌ 将 X_tri 写回 semantic_map
❌ 优化 pose
❌ 优化 point XYZ
❌ 做 shared SE3
❌ 做 object ICP
❌ 做 global BA
❌ 做 surfel update
❌ 做 Gaussian
❌ 训练 adapter
❌ 用 GT 参与 candidate selection
```

GT 只允许用于最后评价。

---

# 13. 输出目录建议

```text
outputs/streaming_couping_v0/pointmap_diagnosis_v2/

    d0_depth_pose_oracle/
        summary.json
        per_frame.csv
        config.json

    d1_pointmap_error_decomposition/
        region_summary.csv
        instance_summary.csv
        confidence_summary.csv
        per_frame.csv
        summary.json

    t0_triangulation_probe/
        anchors.pt
        anchor_summary.csv
        instance_summary.csv
        condition_summary.csv
        controls_summary.csv
        summary.json
```

---

# 14. Codex 最终报告格式

实验结束后生成：

```text
streaming_couping/docs/pointmap_diagnosis_v2_results.md
```

严格按以下结构。

---

## A. D0：Depth × Pose Oracle

回答：

1. `GT depth + GT pose` 是否正确？
2. raw depth 的主要问题是什么？
3. QK pose 是否足够支持 depth reconstruction？
4. direct point head 相对 depth reconstruction 优势有多大？

---

## B. D1：Raw Pointmap Error

回答：

1. boundary 是否显著差于 interior？
2. background 是否更差？
3. 哪些 instance 最差？
4. visible frame count 是否和 error 有关系？
5. point confidence 是否能识别错误？
6. high-confidence-wrong 占多少？

---

## C. T0：Independent Triangulation

回答：

1. 有多少 valid 3D anchors？
2. `X_tri` 是否整体优于 `X_raw`？
3. 是否存在可靠 high-quality subset？
4. 哪些 condition / ray angle / baseline 最可靠？
5. correct persistent ID 是否优于 shuffled control？
6. SAM 的作用是否真正体现在更好的 correspondence / triangulation validity？

---

## D. 下一步 Go / No-Go

只能根据数据选择下面之一。

### GO-T1

如果：

```text
high-quality X_tri
明显优于 X_raw
且
correct ID > shuffled
```

则建议下一阶段研究：

```text
sparse triangulated anchors
→ confidence-weighted residual correction
→ raw pointmap protected fallback
```

本轮不要实现。

### GO-BOUNDARY

如果：

```text
boundary error >> interior
```

但 triangulation 不可靠：

建议下一阶段研究：

```text
SAM boundary-aware geometry / fusion
```

而不是全局 XYZ refinement。

### NO-GEOMETRY

如果：

```text
triangulation 不优于 raw
且
boundary 也不是主要 error source
```

则结论是：

> 当前 frozen SAM3.1 没有提供足够的新 metric geometry 来改善 raw pointmap。

正式方法继续使用：

```text
QK pose + raw full-history pointmap + SAM semantic lifting
```

不要继续堆 training-free pointmap optimizer。

---

# 15. 最重要的研究问题

本轮实验最终只需要回答三句话：

```text
1. raw pointmap 为什么错？

2. SAM 提供的 persistent identity / boundary
   是否正好对应这些错误？

3. 能否通过
   SAM same-instance constraint
   + independent 2D correspondence
   + multi-view rays
   得到比 raw pointmap 更准确的新 3D evidence？
```

只有第三个问题回答“能”，才有充分理由继续设计 pointmap refinement。
