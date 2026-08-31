# SAM3.1 × StreamVGGT：当前 V0 做法与实验结论

更新时间：2026-08-31

## 当前采用的方案

当前仓库恢复并固定为 `V0: dynamic_instance_geometry_baseline`。它是一个可运行的
经验 baseline，不声称 SAM appearance token 已经能够因果地改善相机位姿。

```text
RGB 序列
├─ StreamVGGT
│    └─ raw camera pose + pointmap/local geometry
└─ SAM3.1 forward-only prompt session
     ├─ open-vocabulary prompts → masks
     ├─ persistent object IDs
     ├─ causal late birth → permanent slots
     └─ geometry/quality/static gates
          ↓
    strictly-previous object geometry memory
          ↓
    geometry transport residual
          ↓
    bounded pose correction；无可用成熟实例时 exact camera-baseline fallback
```

具体约束如下：

- SAM3.1 只提供 open-vocabulary prompt 下的 mask、persistent ID 和 slot lifecycle；
  当前不是 class-agnostic 全图分割。示例配置使用 `bed`、`wardrobe`，可按场景修改。
- 不缓存或融合 SAM appearance/local/temporal feature；pose 分支不接收 SAM token。
- 实例可以在首帧之后才出现。birth 帧只建立 observation/memory，不参与同帧 pose；
  只有严格早于当前帧的可靠观测才可形成几何证据。
- 只有 identity、track、geometry、static 条件同时满足的实例才能写入或读取 pose
  memory；moving、unknown 和低质量实例被排除。
- 几何 mask 处理使用冻结的 StreamVGGT raw pose/pointmap，不读 refined 同帧 pose，
  避免同帧循环依赖；结果暂不写回 SAM 内部 memory。
- pose 是冻结 StreamVGGT camera 输出外的 bounded learned residual，因此当前只能称为
  同场景经验 baseline，不能称为显式几何 solver 的理论保证。

默认 V0 配置的时间划分为：

| 用途 | 帧号 | 数量 |
|---|---|---:|
| reference/gauge | 90 | 1 |
| camera baseline 训练 | 105–255，步长 15 | 11 |
| geometry residual 训练 | 270–345，步长 15 | 6 |
| evaluation | 360–525，步长 15 | 12 |

历史 V0 baseline 的复现命令已归档；当前语义地图系统的运行入口是：

```bash
zsh streaming_couping/commands_run_semantic_map.txt
```

不需要先运行历史 V0 baseline；下方的 V0 cache 结果仅用于记录和对比。

主要输出在 `outputs/streaming_couping_v0/`：`baseline_summary.json`、
`frame_diagnostics.csv`、`dynamic_instance_diagnostics.csv` 和 `poses.pt`。

## V0 的实际效果

在一个封存的 V0 场景（30 帧、16 slots）上，冻结 cache 的 raw 分支得到：

| 指标 | raw V0 |
|---|---:|
| tracking IoU | 0.41345 |
| frame IDF1 | 0.50510 |
| pixel IDF1 | 0.78665 |
| voxelIoU@5cm | 0.09964 |
| F-score@5cm | 0.34401 |
| ghost point ratio | 0.25789 |

对应的 GT-mask oracle 为 tracking IoU `0.81773`、frame IDF1 `0.89973`、
voxelIoU@5cm `0.18513`、F-score@5cm `0.50325`。这说明当前主要瓶颈仍然是
mask/instance quality 和下游几何质量；oracle 只用于诊断上界，不属于部署结果。

工程层面，V0 已验证 late birth、persistent slot、成熟度约束、静态实例过滤、
causal prefix 和无可用实例时的 exact fallback。性能结论目前限于同场景/开发审计，
不能据此声称跨场景泛化。

## 已完成实验与效果

下表中的实验均没有修改正式 V0 baseline；除 Real SAM A/B 外，模型没有重新推理。
输出位于服务器 `/data184/open_source/vggtSam/outputs/` 下对应目录。

| 实验 | 设置 | 关键结果 | 决策 |
|---|---|---|---|
| SAM3.1 auto-proposal | 4 clips、每 clip 16 条 class-agnostic visual-point tracks | prompt scope 的 F5cm/voxelIoU 为 `0.12607/0.03728`；auto visual points 为 `0.03167/0.00912`，4 个 clip 无 tracking improvement | NO-GO |
| V0 bidirectional feedback | 单场景 30 帧；depth refinement 与 temporal projection 诊断，未真正反馈 SAM | refined 相对 raw 仅 `voxelIoU +0.00050`、tracking IoU `+0.00025`；删除像素比例仅 `0.000331`；temporal points 未进入 SAM | 不推进 |
| Temporal prompt A–E matrix | 单场景 325 queries；冻结候选，不重跑 SAM | 最佳 `E_depth_gate(rel=0.15)` 的 precision `0.4891`、coverage `0.5784`、coverage F1 `0.5300`，仍不稳定 | 不推进 |
| Real SAM temporal A/B | 单场景 96 queries；single-frame points-only SAM resegmentation | 所有分支低于 raw；mean IoU 为 `0.3598/0.3724/0.3822/0.3762`，raw 为 `0.4135`；恶化帧比例为 `53.3%/43.3%/33.3%/40.0%` | NO-GO |
| Historical-depth Veto | 单场景 30 帧；history median/MAD 或 quantile reference | median/MAD 使 tracking IoU `0.41345→0.35939`，worsened-frame ratio `0.400`；quantile 也退化 | NO-GO |
| Multi-clip affine + ray | 4 场景、120 帧；每 clip 21 calibration + 9 holdout | affine evidence 仅 `1/4`，geometry gate `2/4`；主分支 holdout VoxelIoU `0.00554→0.00428`、F5cm `0.02582→0.01763` | NO-GO |

### Real SAM A/B 详细结果

分支顺序为 `A_center`、`C_surface_5`、`E_rel_0.15`、`E_rel_0.20`：

| branch | mean IoU | frame IDF1 | voxelIoU@5cm | F5cm | worsened-frame ratio |
|---|---:|---:|---:|---:|---:|
| raw V0 | 0.41345 | 0.50510 | 0.09964 | 0.34401 | 0.0% |
| A center | 0.35981 | 0.42025 | 0.06911 | 0.23774 | 53.3% |
| C surface-5 | 0.37241 | 0.43284 | 0.05147 | 0.18872 | 43.3% |
| E rel-0.15 | 0.38218 | 0.45113 | 0.05204 | 0.19121 | 33.3% |
| E rel-0.20 | 0.37619 | 0.44000 | 0.05176 | 0.19134 | 40.0% |

该实验只验证了单帧 point-prompt resegmentation，没有完整 temporal propagation；
不能把它解释为 SAM3.1 所有时序能力均失败，但当前离散点反馈实现已经不适合作为 V0
闭环。

### Multi-clip affine 详细结果

4 个 scene-disjoint clip 共 120 帧，使用前 21 帧 calibration、后 9 帧 holdout。
每个场景独立拟合 positive-slope scale-shift，固定 raw StreamVGGT pose，只替换 raw
SAM object union 内的点。

Depth-head affine 在 4 个场景中只有 1 个同时改善 holdout median 和 P90：

- `00a...`：P90 改善，但 median 略恶化；
- `0a184...`：median 恶化，P90 仅略改善；
- `1a8e...`：median 和 P90 都恶化；
- `1eacc...`：median/P90 都改善。

因此历史 pointmap 不能作为稳定的 affine calibration target。残余误差同时受到
pose、遮挡、动态物体和空间非线性畸变影响，单一 per-scene scale-shift 无法稳定
修复 depth-head。

## 当前结论

当前只保留 V0：

1. 不使用 SAM3.1 class-agnostic auto-proposal。
2. 不使用离散 temporal point prompt 闭环。
3. 不使用历史深度 Veto。
4. 不使用 depth-head affine 或 ray reconstruction correction。
5. 不声称 SAM appearance/token 已经因果改善 pose。

后续任何新方法都必须在独立 scene-disjoint 数据上，与 raw V0 做同协议比较，并同时
报告 tracking、depth、map、恶化帧比例和 fallback/coverage；未通过稳定性 gate 前，
不得写回正式 V0。

## 语义地图 pipeline 实现

在不修改上述 V0 训练和评估路径的前提下，仓库新增了
`src/semantic_mapping/`。它把模型推理与地图融合解耦：

- `GeometryFrame` 是统一几何 contract，支持 `world_points`，也支持未来后端只提供
  `depth + intrinsics + camera_to_world` 的情况；
- `ObjectObservation`/`SegmentationFrame` 是统一的 prompt、实例 ID、mask 和置信度 contract；
- `StreamVGGTGeometryAdapter`、`SAM31SegmentationAdapter` 接入当前模型；V0 cache
  adapters 可以在不重新推理模型的情况下重放；
- `SemanticMapBuilder` 按帧将 mask 投影到世界坐标并融合成体素，同时把动态实例保存到
  独立 track，避免写入静态地图；
- `scene_rgb_map.ply` 和 `scene_semantic_map.ply` 输出所有有效几何经过多帧体素融合后的完整
  场景，后者在场景上下文中覆盖静态 prompt 实例颜色；
- `semantic_map.ply`/`rgb_map.ply` 只输出静态 prompt 物体的融合体素，
  `object_tracks.ply` 则汇总保留 `frame_id` 的逐帧 track 点，不进行体素融合；
- `semantic_map.pt` 和 `map_summary.json` 记录完整场景、语义地图、轨迹及后端、坐标系、尺度
  和对象元数据。

运行入口是 `commands_run_semantic_map.txt`（底层为 `scripts/run_semantic_map.py`）。
当前 RGB 路径仍调用 StreamVGGT + SAM3.1；
切换 HorizonStream 时只需实现同样的 `GeometryProvider`，将输出转换为 `GeometryFrame`，
无需修改地图融合逻辑。当前默认尺度标记为 `unknown`，不会把 StreamVGGT 的 native
坐标自动宣称为米制坐标；该 pipeline 也不启用已判定 NO-GO 的 temporal point prompt、
历史深度 Veto 或 affine correction。
