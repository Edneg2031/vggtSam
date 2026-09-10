# SAM 物体共识位姿反馈实验（object-consensus pose feedback）

更新时间：2026-09-10（实现完成，服务器实验待跑）

## 1. 实验问题

验证：**SAM3.1 的 persistent object tracks 是否可以作为跨帧稳定的 object anchors，
帮助 HorizonStream 检测并修正累计 pose drift，并把修正反馈到后续帧。**

核心思想：

> Persistent objects do not directly provide the correct camera pose. Each
> object proposes a pose residual, while semantic reliability, geometric
> observability, and cross-object agreement determine whether the residual
> should be fed back into the streaming pose accumulator.

不训练任何模型；不修改 HorizonStream backbone / KV / GLA cache；不引入 DINO 或
learned score。

## 2. 数据流（四段式命令）

```text
zsh streaming_couping/commands_run_scannet_object_pose_feedback_100f.txt
```

```text
Stage 1  horizonstream env (cuda:0)
  generate_horizonstream_geometry_cache --save-chunk-cam-maps
  → horizonstream_geometry.pt + horizonizonstream_geometry_chunk_cam_maps.pt

Stage 2a horizonstream env (SAM cuda:2, 优化 CPU)
  run_semantic_map --object-pose-loss-* --object-pose-loss-independent-instances
                   --object-pose-loss-export-feedback-diagnostics
  → raw_pose/ + object_pose_object_only_per_instance/   （Branch D 地图）
  → object_pose_refinement/feedback_diagnostics.pt      （观测/参考云/提案）

Stage 2b 纯 CPU（可单独廉价重跑调阈值）
  run_object_pose_feedback
  → A raw 重放（并断言 ≡ cache 轨迹）
  → B 全部 proposals vs ΔT_GT 逐个评误差
  → C 五个 consensus 变体各自 gating → 注入重放 → 轨迹指标
  → object_pose_feedback/{frame_metrics,object_proposals,consensus_metrics,
     future_pose_gain}.csv + poses.pt + summary.json

Stage 3  evaluate_exported_semantic_map --map-source tracks
  → Branch D 的 raw vs per-instance 物体点云指标

Stage 4  print_object_pose_loss_object_only_metrics + 反馈 summary
```

默认场景 `00a231a370`、帧窗 `90–189`、prompts `bed wardrobe chair rug dustbin`。
两遍式设计：proposals 全部基于 raw 几何计算；SAM 和 refiner 只跑一次，之后所有
consensus/gating/阈值迭代只重跑 Stage 2b。

## 3. 位姿约定（与已验证的 GT feedback POC 完全一致）

| 对象 | 约定 |
|---|---|
| 模型 chunk 输出 / 内部累加器 | w2c，窗口锚定最新帧；累积为左乘 `inv(rel)` 取中值 |
| 公开轨迹 / 评测 | c2w，frame-0 gauge |
| 物体提案 ΔT_k | c2w 左乘：`C[k,t] = T_aligned[k,t] @ inv(T_raw[t])`，同时也是世界点左乘修正 |
| GT 修正 | `ΔT_GT_t = G_t @ inv(R_t)`（同一 gauge） |
| 注入 | 绝对目标 `target_t = ΔT_consensus @ R_t`，经 POC 原语 `_replace_internal_pose_for_public_target` 写入 `online_absolute_poses` 与 `last_absolute_poses` |
| GT 来源 | manifest `world_to_camera` 求逆 → 归一到首个选中帧；**仅在所有反馈决策冻结之后加载** |

注入语义是**绝对目标**而不是增量：anchor（前 5 帧）钉在 raw 世界系上，因此
重复注入是覆盖，不会叠加过修正。`sliding=1` 时注入粒度为逐帧（仅首 chunk 内的
帧以 chunk 边界为粒度）。

## 4. 可靠性、共识与门控

每个 (frame, instance) 提案先算两个规则化置信度（无 learned score）：

- `S_sem`：track_length / visibility 连续率 / mask 像素 / SAM score 加权归一；
  track 过短、可见率过低、score 过低 → 硬拒（`low_track_confidence`）；
- `S_geo`：alignment 改善 / inlier ratio / overlap / 修正幅度 sanity /
  **协方差特征值退化分类**（λ1≥λ2≥λ3 → volumetric/planar/linear/degenerate，
  退化只降权不直接拒）→ 硬拒原因 `degenerate_geometry` /
  `correction_too_large` / `low_geometry_confidence`。

跨物体共识把每个 ΔT_k 转成 `ξ_k = (rotvec, translation)`（与 refiner 优化变量
同参数化的小角度近似），做加权 Huber IRLS；输出 per-object
`translation/rotation_consensus_error` 与 `consensus_inlier`。

五个 ablation 变体一次运行全部算完（重放是 CPU）：

| 变体 | 聚合 | 权重 |
|---|---|---|
| `single` | 最高权重单物体 | S_sem×S_geo |
| `mean` | 加权平均 | 均匀 |
| `robust` | Huber IRLS | 均匀 |
| `robust_semantic` | Huber IRLS | S_sem |
| `robust_semantic_geometric`（主方法） | Huber IRLS | S_sem×S_geo |

帧级门控按优先级输出 reject reason：
`insufficient_objects → low_track_confidence → degenerate_geometry /
correction_too_large / low_geometry_confidence → no_consensus →
correction_too_large（共识幅度）→ no_alignment_improvement`。
最后一个门用 **pairing snapshot 里的真实参考云** 复算加权 NN 残差：consensus
修正统一作用到所有贡献物体的当前世界点后，残差必须严格下降，而不是用
per-instance 最优 loss 代替。所有阈值集中在 `ObjectPoseFeedbackConfig`
（约 40 个参数，全部可通过命令行覆盖）。

## 5. 分支

| 分支 | 内容 |
|---|---|
| A raw | 无注入重放；内置等价性断言（vs cache 轨迹，平移 <1e-3 m / 旋转 <0.01°） |
| B no-feedback | 只分析 proposals（含被拒 best_pose）与 ΔT_GT 的误差及 clamp 饱和标志 |
| C feedback | 五个变体；主方法 `robust_semantic_geometric` |
| D per-instance | Stage 2a/3 的逐实例物体点修正地图（相机位姿保持 raw） |

## 6. 输出

`object_pose_feedback/` 下：

| 文件 | 内容 |
|---|---|
| `object_proposals.csv` | 每提案一行：track/点数/overlap、loss before/after、inlier ratio、特征值与几何类型、ΔT 幅度、S_sem/S_geo、refiner 与 object 级 reject 原因、consensus 误差与 inlier、GT 修正误差与饱和标志 |
| `frame_metrics.csv` | 每 (variant, frame)：raw/feedback 平移与旋转误差、proposal/reliable 数、accept 与 reject 原因、共识幅度 |
| `consensus_metrics.csv` | 每 (variant, frame)：可靠性计数、inlier 数、聚合损失 before/after、GT 共识误差 |
| `future_pose_gain.csv` | 每 (variant, accepted 帧, k=1..10)：raw/feedback 未来误差与增益 |
| `poses.pt` | raw / gt / ΔT_GT / 各变体轨迹 |
| `summary.json` | 8 个问题的数值答案、bottleneck 归因、GO/NO_GO 决策与判据、`refiner_settings_audit` |

RPE 额外报告**排除 correction-boundary 对**的版本（相邻对任一端是修正帧则排除），
避免注入造成的相邻帧 RPE 假跳变。

该边界排除版 RPE 的 translation/rotation 两个 key 由
`object_pose_feedback.RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY` /
`RPE_ROTATION_BOUNDARY_EXCLUDED_KEY` 单点定义，写入方（Stage 2b 重放）和读取方
（`decide_object_feedback`）共用同一常量，避免两端的 key 名漂移。rotation 比值只在
`decisions[*].rpe_rotation_ratio_boundary_excluded_informational` 里作审计输出，
**不进入 `criteria`**，因此不影响 GO/NO_GO。

## 7. GO / NO-GO 判据（仅位姿指标，alignment loss 不进判据）

主方法 `robust_semantic_geometric` 需同时满足：

1. direct ATE 相对 raw 改善 ≥ 5%；
2. accepted 帧的 future translation gain（t+1..t+10）中位数 > 0 且 ≥ 60% 为正；
3. 排边界 RPE translation 不劣于 raw × 1.05；
4. accepted ratio（有提案的在线帧中）≥ 10%。

不满足则 `OBJECT_FEEDBACK_NO_GO`，summary 同时给出启发式瓶颈归因
（`sam_tracking` / `nn_correspondence_or_object_geometry` / `consensus` /
`gating` / `pose_feedback`）。

## 8. 已知边界

- 提案受 refiner clamp 限制（≤10°/0.25m）；`|ΔT_GT|` 超出时用饱和标志区分
  "提案不准" 与 "提案被 clamp"。
- 运动平均的中值会稀释单帧修正（窗口内 9 个候选取中值）；持续多帧注入同一
  方向修正才能稳定传播，这是 accumulator 机制本身的属性，不是 bug。
- 两遍式：proposals 基于raw 几何一次性计算，修正后的世界不回灌给后续提案
  （对 300 帧实验暴露的 reference 污染问题更鲁棒；交织式留作后续实验）。
- `anchor_frame_count` 等反馈配置需与 Stage 2a 的 refiner 参数保持一致（默认 5）。
  该约束现在**强制校验**：Stage 2a 会把 refiner 的 `anchor_frame_count`、
  `max_correction_{rotation_deg,translation_m}`、`max_match_distance_m`、
  `trim_ratio`、`min_matches_per_pair` 写入 `feedback_diagnostics.pt` 的
  `refiner_settings`，Stage 2b 启动时与反馈配置逐项比对（见
  `object_pose_feedback.REFINER_SETTING_MIRRORS`）；不一致直接报错并列出差异，
  结果记录在 `summary.json` 的 `refiner_config_match` /
  `refiner_settings_audit`。确实想用非镜像阈值时传
  `--allow-refiner-mismatch`，降级为 warning 并把该 run 标记为
  `refiner_config_match=false`。
- GT 只在 Stage 2b 所有决策之后加载，summary 记录 `gt_used_for_feedback: false`
  审计标志。

## 9. 相关代码

| 文件 | 角色 |
|---|---|
| `src/semantic_mapping/object_pose_feedback.py` | 配置、S_sem/S_geo、退化分类、consensus、gating、聚合损失、决策与归因 |
| `scripts/run_object_pose_feedback.py` | Stage 2b 入口 + 重放引擎（复用 POC 注入原语） |
| `object_pose_loss_refinement.py` | 新增 `export_feedback_diagnostics`（默认关）：observations / pairing_snapshots / rejected best_pose |
| `generate_horizonstream_geometry_cache.py` | 新增 `--save-chunk-cam-maps`（默认关）：chunk 相机图 aux 文件 |
| `scripts/run_scannet_horizonstream_gt_feedback_poc.py` | 已验证的注入原语与评测函数（本实验原样复用，未修改） |
| `tests/test_object_pose_feedback.py` | 18 个 CPU 测试：Lie 代数、退化分类、可靠性规则、共识拒外点、门控全部 reject 原因、**重放等价性**、注入语义、RPE key 契约、refiner 阈值镜像校验 |

注意 `test_replay_equivalence_with_motion_averaging` 和 `test_replay_injection_semantics`
需要 `horizonstream` 可导入，否则 pytest 会 skip。本地/CI 运行时要显式加上子模块路径：

```bash
PYTHONPATH="$PWD:$PWD/externals/horizonstream" \
  python -m pytest streaming_couping/tests/test_object_pose_feedback.py -q
```
