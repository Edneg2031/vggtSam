
# 语义地图实验路线

更新时间：2026-08-24

## 1. 总原则

实验主线固定为：

    冻结 StreamVGGT geometry
             ↓
    验证/增强 SAM mask 和 persistent identity
             ↓
    验证 object-level semantic map
             ↓
    必要时再做 semantic ownership → geometry feedback

当前不要同时修改 StreamVGGT、SAM prompt、SAM memory、appearance feature 和 map fusion。每次只改变一个模块，并保留 raw、oracle 和错误控制。

当前四个 scene 的实验划分为：

    train: 00a231a370, 0a184cf634
    validation: 1a8e0d78c0
    test: 1eacc65607

这个 2 train / 1 validation / 1 test 是 pilot，不是原计划的 3/1/1 五场景正式 protocol。1eacc65607 已经被 Residual 和 Direct 多次观察，因此不能再把它当 sealed test 来调参。后续结果应标记为 development_audit_only，或者重新建立 scene rotation / 新 test。

## 2. 已完成、不再重复的实验

### V0 semantic-map baseline

已完成：

    raw full-history StreamVGGT pointmap
    + SAM3.1 persistent mask/slot/ID
    + V0 QK pose output

它证明了系统能运行并生成 semantic map，但没有证明 SAM 改善 geometry。

### V6 geometry → SAM

已完成 box + positive geometry support 的 V6 方法，并观察到逐帧 IoU 正提升。后续不应把同一个 IoU 实验重复称为新方法；应该补充 ID 和 map 指标。

### Geometry-only adaptation

Residual 和 Direct PointHead 都已完成：

    Residual test gain: -4.7453% / -5.9564%
    Direct test gain:   -36.0575%

validation 有局部改善，但跨场景 test 失败。当前不再在同一个 test scene 上继续调 loss、learning rate 或 decoder architecture。

### SAM token / memory → pose

已有多轮 SAM local token、memory token、cross-attention、SAM-guided retrieval、shuffled/wrong-ID control 等实验，尚未得到稳定 held-out causal geometry gain。暂时停止继续拼接 SAM token 到 StreamVGGT pose/XYZ head。

## 3. Stage 0：补齐 semantic-map 评价基线

这是下一步最重要、也是成本最低的一步。

固定四个 map branch：

    B0 raw pointmap + raw SAM mask
    B1 raw pointmap + V6 mask
    B2 raw pointmap + GT instance mask       (evaluation-only oracle)
    B3 raw pointmap + V6 mask + map-write gate

GT mask 只用于 offline evaluation，不能进入 candidate generation。

### 必须报告的 2D 指标

    mask IoU / J&F
    tracking recall@0.25 / @0.50
    frame IDF1
    pixel IDF1
    ID switches
    re-entry events / success
    occlusion recovery
    improved-frame ratio
    worsened-frame ratio

### 必须报告的 3D map 指标

    object accuracy
    object completeness
    paired point RMSE
    F-score@5cm / @10cm
    voxel IoU@5cm
    ghost-point ratio
    duplicate-object rate
    object recall@IoU25 / @IoU50

### Stage 0 判别逻辑

    GT-mask oracle 也不好
        → StreamVGGT pointmap/pose 是主要瓶颈

    GT-mask oracle 好，但 raw/V6 SAM 不好
        → SAM tracking/identity 是主要瓶颈

    V6 IoU 好，但 map completeness/ghost 不好
        → 单帧 mask 提升没有转化为 object fusion 提升

    V6 map 好，但 pose 不变
        → 这仍然是有效的 semantic-map 结果，不需要强行修改 pose

GT-mask oracle 的多场景入口现在是：

    zsh streaming_couping/commands_semantic_mapping_multiclip_oracle.txt

它把 GT mask 按 raw-SAM 的 frozen slot assignment 放回同样的 mask 槽位，只在
评估阶段使用 GT，不参与任何候选生成、tracking 或 map-write 决策。这个上界用于
区分：如果 oracle 也不能改善地图，继续改 SAM identity/memory 没有意义；如果
oracle 明显更好，才值得把精力放在 mask ownership 和 re-entry 上。

由于 frozen cache 已经包含 `sam31_online_geometry_compete`，V6 与 oracle 的统一
对照入口是：

    zsh streaming_couping/commands_semantic_mapping_multiclip_v6_oracle.txt

该实验不重跑 SAM/StreamVGGT，只把 raw cache 中的 V6 mask 作为额外 branch，固定
raw branch 做跨分支 assignment。它回答的是：V6 的逐帧 mask 优势，是否真的传递到
IDF1、re-entry 和 object-level map。

## 4. Stage 1：V6 prompt causal ablation

只有在 Stage 0 完成后，才跑 prompt 消融。第一轮只需要：

    S0 raw SAM
    S1 geometry box only
    S2 positive points only
    S3 box + positive points
    S4 shifted geometry control

所有分支必须保持一致：

- 相同 RGB frame sequence；
- 相同 raw SAM tracker；
- 相同 candidate 数量和 SAM threshold；
- 相同 geometry gate；
- 相同 persistent slot；
- 相同 evaluation assignment。

正确几何的基本通过条件是：

    S3 > S0
    S3 > S1
    S3 > S2
    S3 > S4

不能只看 IoU，还要看 IDF1、re-entry、worse-frame ratio 和 object map metrics。

stale geometry 和 equal-count random positive 可以放在第二轮。不要在第一轮一次启动全部分支，因为每个 object/frame 的 SAM prompt session 都有明显成本。

## 5. Stage 2：appearance-assisted identity

如果 Stage 1 显示 V6 改善 mask，却不能改善 IDF1/re-entry，则引入 appearance model。

推荐信息流：

    SAM mask/crop
        ↓
    appearance embedding
        ↓
    object memory
        ↓
    track association / re-entry / duplicate suppression

不要把 CLIP embedding 直接输入 StreamVGGT pose head。

### 模型选择

    CLIP / SigLIP
        更适合 semantic class consistency：chair、desk、cabinet

    DINOv2 / SAM intermediate feature
        更适合同类实例之间的 appearance identity

    StreamVGGT object geometry descriptor
        更适合跨帧 3D shape/position consistency

建议先只选一种 appearance model，不要同时引入 CLIP、DINOv2 和 SAM token。

最小消融：

    A0 geometry only
    A1 appearance only
    A2 geometry + appearance
    A3 geometry + appearance + temporal memory
    A4 shuffled/stale appearance control

CLIP 不能单独被当作可靠的 instance tracker，尤其是场景中存在多个同类别对象时。因此必须有同类多实例和 re-entry subset。

## 6. Stage 3：object-level memory 和 map fusion

V1 persistent 3-D instance memory 已按下面的 geometry-only 版本实现，作为 Stage 3 的第一轮
可运行基线：

    SAM observation
        ↓ active SAM track 直接复用，birth/re-entry 才 association
    persistent object ID
        ↓ center / voxel overlap / Chamfer / category (+ optional appearance)
    accumulated world points

实现入口：

    zsh streaming_couping/commands_semantic_mapping_v1.txt
    zsh streaming_couping/commands_semantic_mapping_v1_evaluation.txt

V0 仍由 `semantic_map.identity_mode: v0_slot` 控制；V1 命令显式使用
`v1_object_memory`，所以两条路线可以在同一个 frozen cache 上复现。V1 输出
`object_memory.json/csv`、`association_events.csv` 和带 `persistent_object_id` 的 PLY。

第一轮默认关闭 appearance，先验证 geometry-only identity memory；只有 V1 的 fragmentation/
re-entry 结果明确显示 identity 仍是瓶颈时，才打开 cached appearance。

先实现显式、可审计的 map memory，不修改 SAM hidden memory：

    short-term memory:
        最近 mask、score、appearance、3D observation

    long-term memory:
        高置信度、无遮挡、几何稳定的 object anchor

    map-write gate:
        track score
        mask area consistency
        geometry confidence
        appearance consistency
        occlusion/re-entry state

比较：

    M0 所有可见 observation 都写入
    M1 只使用 SAM score gate
    M2 score + geometry gate
    M3 short-term / long-term clean anchor
M4 short/long memory + appearance

## 7. V2.3 freeze and multi-clip validation

V2.3 is the current frozen candidate. Before adding V2.4, validate the same
failure-only recovery and confidence-aware memory policy on more than one
clip/scene. The new protocol keeps the raw-SAM assignment fixed separately for
each clip and reports both tracking and map-writing branches:

    raw
    V2.1
    V2.2
    V2.3 tracking + all-visible map writes
    V2.3 tracking + confidence-aware map writes

The all-visible V2.3 branch is an offline control. It uses the frozen V2.3
tracking masks but disables the memory write gate, so a difference between the
two V2.3 rows is attributable to map memory rather than another SAM call.

On the server:

    zsh streaming_couping/commands_plan_semantic_mapping_multiclip.txt
    zsh streaming_couping/commands_run_semantic_mapping_multiclip.txt
    zsh streaming_couping/commands_semantic_mapping_multiclip_ablation.txt

For a single uninterrupted run, use:

    zsh streaming_couping/commands_semantic_mapping_multiclip_all.txt

The first command creates a protocol config and one config per clip. The
second command accepts `SEMANTIC_MAPPING_MULTICLIP_STAGE=cache|baseline|v21|v22|v23|all`
so failed stages can be resumed. The last command is evaluation-only and
writes per-clip CSV rows plus macro mean/std/min/max summaries.

The strict aggregate decision additionally requires V2.3 ghost rate not to
exceed raw. A single development clip should therefore be reported as a
pilot, not as a generalization claim. Do not tune V2.3 thresholds on a sealed
test scene; use the multi-scene result to decide whether a V2.4 memory change
is justified.

The generated multi-scene protocol uses the native raw StreamVGGT pose as the
semantic-map control. QK retrieval is still generated and reported as a
diagnostic, but it is not silently used when its center/rotation gains are
inconsistent across scenes. This prevents the pose ablation from confounding
the SAM recovery and map-memory ablations.

主要目标是：

    ghost-point ratio ↓
    duplicate-object rate ↓
    ID switches ↓
    object completeness ↑
    object voxel IoU ↑
    re-entry success ↑

只有在 re-entry/occlusion 样本足够时，per-object SAM retrack 才有意义。如果一个 episode 没有有效 re-entry event，就不要用它判断 memory 是否有效。

需要明确区分：

    SAM hidden memory
        模型内部 tracker state

    object map memory
        tracking 输出之后的显式 map-write/read state

后者不能声称是 SAM3-DMS 的原样复现。可以把独立 forward session per object 作为 per-object retrack ablation，但必须这样命名。

## 7. Stage 4：可选的 geometry feedback

只有当下面条件都满足时才启动：

- V6 或 appearance-enhanced SAM 的 identity 指标稳定改善；
- object map completeness/ghost 指标改善；
- GT-mask oracle 表明 geometry 仍有明显误差空间。

反馈先从不训练的显式控制开始：

    G0 raw StreamVGGT
    G1 dynamic-object exclusion
    G2 background confidence weighting
    G3 background + multiple static-object support
    G4 instance-local-only negative control

反馈不直接修改原始 mask 或隐式 token，而是形成显式 correspondence/static factor。

评价必须同时覆盖：

    pose
    pointmap RMSE / Median / P90
    object map accuracy/completeness
    semantic map ghost/duplicate rate

如果只有 pose 变好、semantic map 没变好，不能把它称为 semantic-map improvement。

## 8. Stage 5：独立 geometry memory

RetrieveVGGT / FrameVGGT 风格的 memory 是另一条支线，不要和 SAM token fusion 混在一起。

最小协议：

    full-history
    Top-4
    recent K
    diverse K
    segment sampling K=16/24/32
    frame-block memory

需要保持完整 frame/block，不把单个 token 采样结果当作完整 frame memory。当前 QK Top-4 不能被视为 RetrieveVGGT 的完整复现。

## 9. 明天的实际执行顺序

### 第一步：离线评价已有结果

先确认服务器已有 V0/V6 cache 和 output，再计算：

    IDF1 / ID switches / re-entry
    GT-mask oracle map metrics
    raw SAM map metrics
    V6 SAM map metrics

这一步不应重新训练 Residual 或 Direct。

### 第二步：小规模 prompt ablation

在少量 scene/episode 上运行：

    raw / box-only / points-only / box+points / shifted

先确认 candidate budget、显存和运行时间，再决定是否扩展到全部 episode。

### 第三步：object map write gate

在同一批 masks 上 offline 比较 M0–M3。它主要是 CPU/point-cloud 评价，成本远低于重新运行 SAM。

### 第四步：按判别结果选择 CLIP 或 DINOv2

    IDF1/re-entry 差
        → appearance model / memory

    IDF1 好但 map completeness 差
        → map fusion / geometry

    oracle map 也差
        → geometry memory / pose branch

## 10. 当前停止条件

以下实验暂时停止：

    Residual/direct 在同一 test scene 上继续调参
    SAM token 直接接 pose/XYZ
    把不同 memory、token、geometry decoder 一次性混合
    只凭单帧 IoU 宣称 semantic-map 提升

当前最有价值的论文证据不是再增加一个 decoder，而是完整展示：

    raw SAM → V6 geometry-guided SAM
            → identity/re-entry
            → object-level semantic map

并用 GT-mask oracle、wrong geometry 和 appearance shuffle controls 说明提升究竟来自哪里。
