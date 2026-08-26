# DINOv3 Persistent Object Feature → Object Geometry

## 0. 当前目标

这是一个项目验证，不是为了先写论文。希望确认下面这条链路是否真的能够改善语义地图中的 object-level geometry：

```text
RGB stream
   ├── StreamVGGT
   │      ├── geometry feature F_geo
   │      └── raw world pointmap X_raw
   │
   └── SAM3.1
          ├── object masks
          ├── persistent track IDs
          └── track scores
                    ↓
              DINOv3 dense patch feature
                    ↓
              mask-based object pooling
                    ↓
              single-frame feature e_i,t
                    ↓
              track-wise temporal aggregation
                    ↓
              persistent object feature E_i,t
```

最初的假设是：

> SAM3.1 提供 object correspondence，DINOv3 提供稳定的 object appearance representation；跨帧 persistent appearance 可能帮助 StreamVGGT 改善 object-level point cloud。

需要特别区分：

1. DINOv3 object feature 是否比 geometry-only 有用？
2. persistent aggregation 是否比当前帧 feature 有用？
3. 正确的 track identity 是否比 shuffled identity 有用？
4. 改善是否真正传递到最终 world-space semantic map？

---

## 1. 当前数据和运行协议

当前使用四个 scene-disjoint clip，每个 clip 30 帧：

```text
train:
  00a231a370_uniform30_r1
  0a184cf634_uniform30_r1

validation:
  1a8e0d78c0_uniform30_r1

test:
  1eacc65607_uniform30_r1
```

数据来源：ScanNet++ processed pinhole 2D 数据。

当前运行协议：

```text
StreamVGGT: frozen
StreamVGGT PointHead/CameraHead: frozen
SAM3.1: frozen
DINOv3: frozen
trainable component: small residual geometry head only
```

DINOv3 checkpoint：

```text
/home/bod/86Nas/95_data_bak/FoundationModels/dinov3/dinov3-vitl16
```

该目录包含 Hugging Face 格式的 `config.json` 和 safetensors 权重。实际实验使用 ViT-L/16，输出维度为 1024。

DINO cache：

```text
/data184/open_source/vggtSam/outputs/streaming_couping_dinov3_object_features
```

四个 cache 均已成功生成：

```text
00a231a370_uniform30_r1.pt  valid=118
0a184cf634_uniform30_r1.pt  valid=53
1a8e0d78c0_uniform30_r1.pt  valid=84
1eacc65607_uniform30_r1.pt  valid=75
```

这里的 `valid` 是有效的 mask-pooled object feature 数量，不是全部 30×16 个 slot 都有有效 DINO feature。

---

## 2. 当前实现的 DINO feature pipeline

### 2.1 Dense feature

DINOv3 输出 dense patch feature：

```text
F_dino[t] ∈ R^(Hf × Wf × 1024)
```

SAM mask 被 resize 到 DINO patch grid，在 mask 内做 mean pooling：

```text
e_i,t = MaskedMeanPool(F_dino[t], M_i,t)
```

最后做 L2 normalization。

没有使用 CLS token，也没有加入 attention pooling 或 cross-attention。

### 2.2 Persistent feature

对于同一个 SAM track slot，采用 causal EMA：

```text
E_i,t = β E_i,t-1 + (1 - β) e_i,t
β = 0.9
```

保留三个版本：

```text
single_features
persistent_features
shuffled_persistent_features
```

`shuffled_persistent_features` 将 slot feature 做固定随机置换，再执行相同的 temporal aggregation，用来控制“只是增加了历史 feature”这一因素。

### 2.3 目前代码中已经修复的问题

早期 checkpoint selection 使用了 SAM support，而最终 object metric 使用 GT support，二者评估 population 不一致。现在已经修改为：

```text
validation checkpoint selection:
GT object support paired RMSE
then GT object support P90
```

test 只在所有 residual head checkpoint 冻结后打开。

---

## 3. 当前 geometry head

当前 head 的输入和输出是：

```text
F_geo(p) + projected object feature E_i
                 ↓
          small residual decoder
                 ↓
              ΔXYZ(p)
```

最终：

```text
X_new(p) = X_raw(p) + ΔXYZ(p)
```

修正只写在 SAM mask union 内：

```text
SAM object region: learned residual
background: exact raw fallback
```

比较了五个分支：

```text
raw
geometry_only
single_view_dino
persistent_dino
shuffled_persistent_dino
```

其中：

- `geometry_only`：只使用 StreamVGGT geometry feature；
- `single_view_dino`：使用当前帧 DINO object feature；
- `persistent_dino`：使用基于 SAM track identity 的 EMA feature；
- `shuffled_persistent_dino`：错误 track identity 的控制分支。

所有大模型都 frozen，只训练 residual head。

---

## 4. 第一轮 object geometry 结果

### 4.1 Validation scene

```text
branch                         object RMSE       object F-score@5cm
raw                            0.20261            0.44985
geometry_only                  0.20522            0.37689
single_view_dino               0.20609            0.38548
persistent_dino               0.20676            0.36965
shuffled_persistent_dino      0.20559            0.40016
```

validation 上 raw 明显优于所有 learned residual branch。

### 4.2 Test scene

```text
branch                         object RMSE       object F-score@5cm
raw                            0.47011            0.11466
geometry_only                  0.42488            0.12754
single_view_dino               0.42396            0.11300
persistent_dino               0.41798            0.13442
shuffled_persistent_dino      0.44202            0.11459
```

test 上：

- geometry-only 比 raw 好；
- single-view DINO 的 RMSE 略好于 geometry-only，但 F-score 较差；
- persistent DINO 的 RMSE 和 F-score 都优于 geometry-only；
- shuffled persistent 明显差于正确 persistent。

这说明 test scene 存在一个有价值的信号：

```text
persistent_dino > shuffled_persistent_dino
```

但这个信号没有在 validation scene 上稳定出现，因此目前不能认为已经证明了 persistent identity 的泛化价值。

---

## 5. 当前 semantic-map readout 结果

当前 geometry experiment 使用相同 frozen raw SAM masks 对 corrected pointmap 做 map evaluation。这个 readout 还不是完整重新设计的 semantic memory pipeline。

### 5.1 Validation scene

```text
branch                         map voxelIoU        map F-score@5cm       ghost
raw                            0.08063              0.28214               0.11424
geometry_only                  0.02365              0.09047               0.21815
single_view_dino               0.04490              0.13600               0.22611
persistent_dino                0.03346              0.10802               0.22449
shuffled_persistent_dino       0.05286              0.17334               0.17278
```

### 5.2 Test scene

```text
branch                         map voxelIoU        map F-score@5cm       ghost
raw                            0.01906              0.05944               0.26118
geometry_only                  0.01895              0.07637               0.20885
single_view_dino               0.01580              0.06474               0.23919
persistent_dino                0.02023              0.08686               0.20811
shuffled_persistent_dino       0.01554              0.06464               0.25519
```

test 上 persistent DINO 的 map F-score 和 voxelIoU 略好于 raw/geometry-only，ghost 也较低；但 validation 上所有 learned residual branch 都严重破坏 map。

因此当前 aggregate decision 是：

```text
NO_GO
```

当前方法不能作为稳定的部署方案。

---

## 6. V2.4 causal world-space gate

为了减少 learned residual 对 map 的污染，增加了一个不使用 GT、不使用未来帧的 inference-time gate。

### 6.1 Hard gate

gate 使用：

```text
correction magnitude
历史 object centroid
历史 robust extent
当前 raw/candidate 的 world-space consistency
point confidence
SAM track score
```

只有满足以下条件才接受 residual：

```text
correction q90 足够小
candidate 不明显比 raw 更偏离历史 object geometry
```

### 6.2 Soft gate

soft gate 不直接丢弃整个 observation，而是使用连续权重：

```text
X = X_raw + w_residual × w_memory × (X_candidate - X_raw)
```

### 6.3 V2.4 test 结果

```text
branch                         object RMSE       map F-score@5cm       voxelIoU       ghost
raw                            0.47011            0.05944               0.01906        0.26118
persistent ungated             0.41798            0.08686               0.02023        0.20811
persistent hard-gated          0.47013            0.05944               0.01906        0.26118
persistent soft-gated          0.46961            0.06076               0.01913        0.25407
```

hard gate 只接受：

```text
2 / 52 observed object observations
```

soft gate：

```text
high_trust=2
attenuated=15
suppressed=35
weighted_point_ratio=0.09930
```

### 6.4 V2.4 结论

gate 可以显著降低 residual 对 map 的污染，但同时删除了大量可能有效的 correction。因此它解决的是：

```text
如何少犯错
```

没有解决：

```text
如何稳定地产生正确 geometry correction
```

V2.4 最终仍然是：

```text
NO_GO
```

---

## 7. 当前最重要的判断

当前结果不应解读为“DINOv3 没用”，更准确的表述是：

> DINOv3 object appearance feature 被放到了一个不合适的职责上：它被要求直接帮助预测每个像素在世界坐标中的 `ΔXYZ`，但它本身不包含足够的 metric geometry 信息。

DINOv3 能提供的主要是：

- object appearance；
- category/identity similarity；
- 对视角和局部纹理变化的语义稳定性；
- 可能的 object re-identification signal。

DINOv3 不能直接提供：

- 每个像素的深度；
- 前后表面关系；
- 遮挡边界的真实 3D 位置；
- object 的绝对世界坐标；
- 可靠的 surface completion。

所以当前这条链：

```text
DINO feature → broadcast to mask → per-pixel ΔXYZ
```

存在明显的 representation-to-target mismatch。

---

## 8. 当前方法的具体局限

### 8.1 Appearance feature 与 metric geometry 不匹配

同一个 chair 在不同视角下的 DINO feature 可能相似，但这并不能决定当前像素应该移动到哪个 3D 位置。

EMA 还可能把不同视角下的 appearance 混合成一个过于平滑的 vector，丢失当前 view-specific information。

### 8.2 object-level feature 被广播到整个 mask

同一个 `E_i` 被广播到 object mask 内所有 patch/pixel。

因此它没有直接表达：

- 物体哪个部分是前表面；
- 哪个部分属于边界；
- 哪些区域被遮挡；
- 不同表面是否应该有不同修正。

虽然 `F_geo(p)` 保留了局部 geometry feature，但 DINO feature 只提供一个全局 object condition，二者的对应关系仍然很弱。

### 8.3 训练目标与最终 map 目标不一致

当前训练主要优化 point coordinate loss：

```text
point error / robust point loss
```

最终 semantic map 关心的是：

```text
voxel occupancy
multi-view consistency
completeness
ghost rate
object-level F-score
```

point RMSE 下降并不保证 voxel map 变好。validation 已经明确展示了：object residual 可能看起来合理，但 map occupancy 被破坏。

### 8.4 当前 persistent memory 其实只是 appearance EMA

现在的 persistent 主要是：

```text
SAM track ID → DINO feature EMA
```

它不是：

```text
SAM track ID → world-space observations → persistent 3D object memory
```

真正能改善 semantic map 的 memory 应该维护：

- object world-space centroid；
- voxel occupancy；
- point confidence；
- observation count；
- view diversity；
- object appearance；
- 当前 observation 与历史 object 的 correspondence。

### 8.5 当前 gate 假设过强

当前 gate 默认：

```text
correction 越大越不可信
candidate 越接近历史 centroid/extent 越正确
```

但一个系统性偏差的正确修正可能本来就比较大；不同视角看到的 object surface 也可能导致 centroid/extent 变化。因此 hard gate 容易把有效 correction 一起拒绝。

### 8.6 数据规模不足

当前只有：

```text
2 train scenes
1 validation scene
1 test scene
```

所以当前结果最多说明：某个 test scene 上存在 persistent DINO 的正向信号；不能说明已经稳定泛化。

### 8.7 当前 map evaluation 还没有完全体现 object memory

当前 DINO geometry map readout 使用：

```text
same raw SAM masks + corrected pointmap
```

它还不是一个完整的 persistent object voxel fusion pipeline。因此 residual 结果和 semantic-memory 结果不能完全等同。

---

## 9. 已增加的下一步诊断

已经增加一个只做分析、不训练、不重跑模型的命令：

```bash
zsh streaming_couping/commands_analyze_dinov3_residual_calibration.txt
```

它复用现有 frozen cache 和 `models.pt`，分析：

- correction magnitude 分桶；
- raw 与 candidate 的同点误差变化；
- correction direction 与真实 target residual 的 alignment；
- SAM 覆盖区域与全部 GT object 区域；
- boundary/interior；
- DINO feature valid/invalid；
- track age；
- 每个 object 的改善比例。

输出：

```text
/data184/open_source/vggtSam/outputs/streaming_couping_dinov3_residual_calibration/copyable_result.txt
/data184/open_source/vggtSam/outputs/streaming_couping_dinov3_residual_calibration/correction_bins.csv
/data184/open_source/vggtSam/outputs/streaming_couping_dinov3_residual_calibration/condition_summary.csv
/data184/open_source/vggtSam/outputs/streaming_couping_dinov3_residual_calibration/object_calibration.csv
```

这个诊断的目的不是产生一个新的 GO/NO-GO，而是判断：

1. correction magnitude 是否真的能作为 trust signal；
2. residual 是否主要在 boundary/occlusion/re-entry 条件下有价值；
3. DINO feature validity 是否与收益相关；
4. persistent 的收益是否真的来自 identity，而不是 feature smoothing 或 sample selection。

---

## 10. 推荐的下一版方法方向

目前最有动机的方向不是继续把 DINO feature 直接接到 `ΔXYZ`，而是改变 DINO 的职责。

### 10.1 DINO 用于 correspondence / retrieval

新的职责链：

```text
SAM mask + current StreamVGGT world points
                 ↓
          current observation
                 ↓
       DINO appearance feature
                 ↓
  compare with persistent object appearance
                 ↓
     identity / re-entry confidence
```

DINO 用来回答：

```text
当前 observation 是否属于历史 object i？
```

而不是直接回答：

```text
当前每个像素的 XYZ 应该是多少？
```

### 10.2 StreamVGGT 提供 geometry，world memory 提供 persistent map

建议的主链：

```text
StreamVGGT raw world pointmap
          + SAM persistent mask/track
          + DINO appearance correspondence
                    ↓
          persistent object registry
                    ↓
       object-level world-space voxel memory
                    ↓
       confidence-weighted multi-view fusion
                    ↓
             semantic object map
```

每个 object memory 可以维护：

```text
object_id
category
appearance prototype
world centroid
world extent/covariance
voxel occupancy
per-voxel confidence
observation count
last seen frame
view diversity
```

### 10.3 Association score

一个更合理的 observation-to-object score 可以综合：

```text
projection overlap
world centroid distance
voxel overlap / Chamfer similarity
DINO cosine similarity
category consistency
track score
point confidence
```

其中 geometry 和 projection 应该是主要约束，DINO 是 identity/re-identification 的辅助信号，而不是 geometry 的唯一来源。

仓库中已经存在 projection-first object memory 的基础实现：

```text
streaming_couping/src/projection_object_memory.py
```

它已经支持：

- historical world points 投影到当前帧；
- mask projection overlap；
- centroid/extent/chamfer/voxel association；
- temporal confirmation；
- persistent object registry；
- object-level voxel fusion；
- appearance term。

下一步可以把 DINO object feature 作为 appearance term 接入这个 memory，并在四个 clip 上比较：

```text
raw all-visible map
projection/world-memory without DINO
projection/world-memory + single-view DINO
projection/world-memory + persistent DINO
projection/world-memory + shuffled DINO control
```

这个实验不需要重新训练 residual head，更直接对应最终 semantic-map 目标。

---

## 11. 如果仍然要保留 residual，应该如何改

只有当 calibration 证明 residual 在某些条件下有稳定正收益时，才保留 residual。

建议改成：

```text
F_geo(p)
+ local object geometry features
+ object appearance E_i
+ world-memory retrieval features
+ confidence/track-age/view-diversity features
        ↓
calibrated residual/trust head
        ↓
small bounded correction
```

而不是只有：

```text
F_geo(p) + global E_i → ΔXYZ(p)
```

训练目标也应至少包含：

```text
point coordinate loss
object centroid consistency
surface/voxel consistency
multi-view same-object consistency
correction magnitude regularization
```

推理阶段使用 learned trust score：

```text
w_i,t = calibrated confidence
X_new = X_raw + w_i,t ΔXYZ
```

`w_i,t` 应该由可观测 runtime features 预测，而不是由 GT 或固定的 `0.08m` 阈值决定。

---

## 12. 当前不建议做的事情

暂时不建议：

- 直接换 DINOv3 ViT-7B；
- 继续调 `max_correction_m=0.08`；
- 继续叠加 hard/soft gate；
- 解冻 StreamVGGT；
- 解冻 DINOv3；
- 加入 CLIP、SAM hidden token、cross-attention；
- 在只有一个 test scene 的情况下宣称 persistent identity 已经有效；
- 用 global RMSE 单独判断 semantic map 是否改善。

---

## 13. 希望 GPT 重点讨论的问题

请重点回答下面这些问题：

### A. 关于表示和职责

1. DINOv3 dense patch feature 最适合在这个系统中承担什么职责？
2. 对于 object geometry，DINO 应该用于 correspondence、retrieval、confidence，还是仍然值得用于 residual conditioning？
3. 如何避免把 appearance invariance 错误地解释成 metric geometry？

### B. 关于 persistent memory

1. 对于 SAM track + StreamVGGT world pointmap，什么样的 object memory 最可能改善 voxel occupancy 和 completeness？
2. appearance EMA 是否会损失 view-specific geometry information？
3. 是否应该同时维护：

   ```text
   appearance prototype
   current-view appearance
   world-space voxel memory
   observation history
   ```

4. 如何用 DINO similarity 辅助 re-entry，但避免错误 identity 把两个相似 object 合并？

### C. 关于 geometry correction

1. 是否有比直接预测 world-space `ΔXYZ` 更合理的 residual parameterization？
2. 是否应该预测 depth/ray residual、object-centric coordinate、surface offset 或 confidence，而不是直接预测 XYZ？
3. 如何构造与最终 map metric 一致的训练 loss？
4. 如何从历史 world-space object memory 生成当前帧可用的 geometric target？

### D. 关于实验设计

1. 在当前只有 2 train / 1 validation / 1 test scene 的条件下，最小但有信息量的下一组实验是什么？
2. 如何区分 persistent identity 的真实作用、feature smoothing 的作用，以及更多历史观测的作用？
3. 应该如何设计 shuffled、delayed、single-view、world-memory-only 等 controls？
4. 对 semantic map 项目来说，object RMSE、F-score、voxelIoU、ghost rate、completeness 应该如何排序？

---

## 14. 当前暂定结论

目前最稳妥的结论是：

```text
当前 DINOv3 residual geometry 方法：NO_GO
```

但这不等于：

```text
DINOv3 无用
```

更合理的下一步是先完成 residual calibration，然后优先验证：

```text
DINO persistent feature
→ object correspondence / re-identification
→ world-space persistent object memory
→ confidence-weighted voxel fusion
```

在这个版本中，StreamVGGT 负责提供 geometry，SAM3.1 负责 mask/track，DINOv3 负责 appearance identity，memory 负责跨视角 world-space accumulation。这个职责分工比让 DINO 直接预测 metric XYZ 更符合各模块实际能力。

---

## 15. 已落地的 DINO patch-geometry feasibility diagnostic

针对上面的新方向，已经增加：

```text
streaming_couping/scripts/analyze_dinov3_patch_geometry_retrieval.py
streaming_couping/commands_analyze_dinov3_patch_geometry_retrieval.txt
```

用户在服务器上只需要执行：

```bash
zsh streaming_couping/commands_analyze_dinov3_patch_geometry_retrieval.txt
```

命令内部固定 protocol、DINOv3 ViT-L/16、服务器 Python、GPU 和所有 output path；只刷新 DINO dense feature cache，不重跑 StreamVGGT/SAM，不训练 residual head。

### 15.1 候选生成

每个当前 patch 只能检索严格更早的 frame。候选模式为：

```text
correct_track_dino    同一 SAM persistent track ID 内的 DINO 检索
random_same_object    同一 track 内的随机历史 patch
shuffled_track_dino   错误 track ID 内的 DINO 检索
unrestricted_dino     所有历史 object patch 内的 DINO 检索
```

DINO 检索默认加入 mutual-nearest-neighbor 过滤，并使用 point confidence 和 SAM track score 过滤无效 patch。候选生成阶段不会读取 GT；脚本还会把 evaluation target 暂时隐藏，所有候选冻结后才恢复 target pointmap 并打开 GT mask。

### 15.2 两种 geometry 判据

每个 retrieval 同时报告：

1. `object_surface`：历史点到当前 GT object 的聚合 surface cloud 的最近距离。它回答历史 observation 是否至少落在同一物体表面附近。
2. `query_surface`：历史点到当前 query pixel 的 GT world point 的距离。它更严格，只有历史 patch 真正对应到当前局部表面时才会变好。

因此不能只看 object-surface gain。判断 DINO 局部 correspondence 是否有价值时，应优先看：

```text
query_surface retrieved RMSE
query_surface P90
query_surface improved fraction
coverage
```

object-surface 指标只适合作为较宽松的 object-level support 参考。

### 15.3 输出

结果目录：

```text
/data184/open_source/vggtSam/outputs/streaming_couping_dinov3_patch_geometry_retrieval/
```

包含：

```text
summary.json
match_metrics.csv
similarity_bins.csv
per_object_metrics.csv
per_scene_metrics.csv
paired_comparison.csv
condition_summary.csv
copyable_result.txt
```

`paired_comparison.csv` 会在各 control 与 `correct_track_dino` 的共同 query 集合上重新统计，避免某个分支只匹配到容易样本而造成虚假的优势。当前代码只做 feasibility analysis：不写入新的 XYZ，不训练、不选择 checkpoint，也不把 GT 参与 correspondence。只有当 `correct_track_dino` 在严格 query-surface 指标上同时优于 raw、random same-object、shuffled-track，并且覆盖率和 P90 没有明显恶化时，才值得继续实现 DINO-keyed geometry memory 或 DINO reprojection consistency。
