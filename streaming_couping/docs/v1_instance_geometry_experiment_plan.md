# StreamVGGT + SAM3 V1：实例局部几何实验计划

## 1. 实验目标

在不降低 V0 整体点云质量、不改动 V0 QK 位姿的前提下，验证 SAM3 persistent instance
是否能帮助改善局部物体点云。

核心机制不是让 SAM 预测三维位移，而是：

- SAM mask 决定哪些点属于可调整的实例区域；
- persistent ID 决定哪些跨帧观测属于同一物体；
- StreamVGGT 多帧 world pointmap 决定点的调整方向和大小。

场景按静态世界处理，StreamVGGT 与 SAM3.1 均冻结，不训练新网络。

## 2. 固定输入与边界

- 位姿：保留 V0 的 first-frame anchor + native QK Top-4 camera pose；
- 点云初值：保留 full-history StreamVGGT raw world pointmap；
- 实例信息：SAM3.1 persistent mask、slot、track ID 与 track score；
- 当前 prompt：`wardrobe / chair / picture / picture frame / mat / rug / table / desk /
  cabinet / nightstand / shelf / lamp`；
- `bed` 不参与，因为大面积实例会接近全局场景约束，削弱局部实例实验的可归因性；
- 最多同时使用 5 个成熟实例，但 SAM registry 继续保留 16 个永久 slot；
- GT 只能在候选点云完全生成后用于评分，不能参与实例筛选、匹配或位移计算。

V0 是不可覆盖的 fallback。任何 V1 分支失败时，正式输出仍然使用 V0 raw world pointmap。

## 3. 第一步：重新建立无 bed 的观测基线

服务器运行（提示词改变后必须重建 SAM/StreamVGGT 联合 cache；QK pose artifact 可以复用）：

```bash
BASELINE_REBUILD_CACHE=1 zsh streaming_couping/commands_v0_baseline.txt
```

QK pose 只依赖 RGB，可以复用已有 artifact；本次强制重建 cache 是为了重新运行十二类 prompt 的
SAM tracking。重点检查：

- `semantic_map/prompt_discovery.csv` 中每个 prompt 的 raw / eligible / retained 数量；
- `semantic_map/copyable_result.txt` 中的 `discovered_track_count`；
- 每个 track 的 `prompt / visible_frames / saved_points`；
- `semantic_map/tracks.csv` 是否至少包含两个可重复观测的局部物体。

### 实例成熟条件

第一版固定使用以下条件，不根据 GT 调参：

- 历史可见帧数至少 4；
- SAM track score 至少 0.50；
- erosion 后 mask 面积占图像的 0.5%–25%；
- 当前帧有效 interior point 至少 256；
- StreamVGGT normalized confidence 至少 0.30。

如果整条序列少于两个成熟实例，先停止几何实验。此时应该增加合适的小型静态物体 prompt
或更换包含更多物体的序列，而不是调整几何优化器。

### 3.1 当前无 bed 基线结果

2026-08-17 的 30 帧运行发现两个 track：

| slot | prompt | birth frame | visible frames | semantic-map saved points |
|---|---|---:|---:|---:|
| 0 | wardrobe | 90 | 22 | 55,765 |
| 1 | chair | 135 | 14 | 23 |

整体语义覆盖为 13.947%。当前结果只证明 SAM 找到了两个跨帧 track；`saved_points=23` 来自语义地图
40 万点全局置信度抽样，不能据此断言 chair 的原始 mask 只有 23 个点。I0 必须读取 cache 中未抽样的
dense mask、confidence 与 pointmap，再决定 chair 是否满足 256 个 interior point 的成熟门槛。

## 4. I0：Instance Geometry Feasibility Audit

I0 不修改点，只回答：正确 persistent ID 是否组织出了更一致的跨帧三维观测？

对每个当前实例，只使用 `frames < t` 的历史 raw world points，计算：

- 当前点到同 ID 历史点/局部表面的距离 median 与 P90；
- 局部 surface thickness / normal-direction variance；
- 有效匹配比例；
- 每个实例的历史帧数、点数、mask 面积和置信度。

比较三种 ownership：

1. `correct_id`：真实 persistent ID；
2. `shuffled_id`：保持点数和 mask 不变，只打乱跨帧 ID；
3. `shifted_mask`：保持 mask 面积近似不变，但移动空间区域。

### I0 Go 条件

- 至少两个成熟实例参与；
- `correct_id` 的跨帧最近距离和法向残差同时优于 shuffled 与 shifted；
- 优势不能只来自单个实例或单帧。

Surface thickness 继续记录为诊断，但不作为 I0 硬门槛：错误区域也可能恰好是很薄的平面，单独
依赖 thickness 不能证明 persistent ID 正确。

如果 I0 不通过，停止 SAM 几何优化主线：SAM 保留 semantic lifting，不声称改善 pointmap。

## 5. I1：Causal Bounded Instance Surfel Adjustment

只有 I0 通过才生成修改后的候选点云。

对当前实例点 `x`，从相同 persistent ID 的历史 raw points 建立局部 surfel：中心 `c`、法向
`n`。只沿表面法向做有界调整：

```text
r  = n^T (x - c)
x' = x - clamp(alpha * r, -delta, delta) * n
```

第一版约束：

- 只调整 erosion 后的 mask interior；
- mask boundary、背景、未成熟实例保持逐点 exact raw；
- surfel 必须包含至少 3 个不同历史帧的支持；
- 只采用局部平面性和匹配距离均合格的 surfel；
- 位移 `delta` 按场景尺度固定并在所有分支共享；
- 历史 state 始终由 raw points 构建，不将修改后的点写回 memory；
- 不修改 camera pose，不运行 ICP、BA、3DGS 或可训练 residual network。

## 6. I1 对照分支

| 分支 | 几何处理 | SAM 信息 |
|---|---|---|
| `raw` | 不调整 | 不使用 |
| `global_surfel` | 全局使用相同 surfel 调整 | 不使用实例 |
| `foreground_union` | 只在所有 mask 的并集内调整 | 使用 mask，不使用 ID |
| `correct_id` | 按实例历史 surfel 调整 | 正确 mask + persistent ID |
| `shuffled_id` | 相同算法与点数 | 错误跨帧 ID |
| `shifted_mask` | 相同算法与点数 | 错误空间 ownership |

所有分支必须使用相同候选点数、置信度门槛、位移上限和评分 alignment。

## 7. 评分指标

### 整体地图下限

- paired weighted point RMSE；
- fused symmetric mean / Chamfer 类距离；
- 整体 worse-frame 数量。

### 实例主要指标

- mature-instance paired RMSE；
- per-instance symmetric distance；
- surface thickness / normal variance；
- 每个实例的改善率和 worse-frame 数量。

### 安全与可归因指标

- background exact-raw rate，目标为 100%；
- modified point fraction；
- displacement median / P95 / maximum；
- correct、shuffled、shifted 的 equal-support 审计；
- candidate generation GT fields 必须为 0。

## 8. I1 验收条件

只有同时满足以下条件，才能声明 SAM 帮助了局部 pointmap：

1. `correct_id` 的整体点云指标不低于 raw；
2. mature-instance 几何优于 raw；
3. 至少两个实例获得正收益；
4. `correct_id` 同时优于 global、union、shuffled 和 shifted；
5. 背景点保持 exact raw，所有位移均满足预设上限。

结果解释：

- `correct_id ≈ global/union`：普通局部融合有效，但 SAM identity 没有新增贡献；
- `correct_id ≈ shuffled/shifted`：优化器在平滑点云，persistent association 没有被利用；
- instance 改善但 global map 下降：不能部署，继续使用 V0；
- correct 稳定最好且整体不下降：再考虑 memory writeback 或低自由度可训练 residual；
- I1 通过前不实现 3DGS、joint pose–geometry 或复杂网络。

## 9. 实际执行顺序

1. 运行无 `bed` 的 V0 baseline，确认新 tracks；
2. 查看 `copyable_result.txt` 和 `tracks.csv`，先做实例数量门控；
3. 实现并运行一个统一的 I0 + I1 命令；
4. I0 失败时命令只输出 audit，不生成正式 candidate；
5. I0 通过时同一次运行继续生成六个 I1 分支并评分；
6. 根据固定验收条件决定是否保留 `correct_id` candidate；
7. 无论结果如何，V0 raw pointmap 始终保留且不会被覆盖。

统一命令：

```bash
zsh streaming_couping/commands_v1_instance_geometry.txt
```

命令先运行 I0。I0 不通过时安全停止并只输出审计；I0 通过时在同一次运行中继续 I1 六分支评分。
