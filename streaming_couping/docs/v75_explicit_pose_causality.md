# V7.5：动态实例显式位姿因果实验

V7.5 不是新的 token fusion，也不训练 pose head。它只回答两个问题：

1. SAM3.1 的真实实例区域是否比全图、外接框、同面积随机区域和过期 mask 更适合约束
   StreamVGGT 相机中心？
2. 正确的 persistent instance 历史是否比当前帧、错误实例 ID 和无历史更有效？

## 一键运行

```bash
zsh streaming_couping/commands_v75_explicit_pose_causality.txt
```

命令会先严格审计 V7.4 的几何、位姿、mask 和 identity cache；V7.5 不要求也不读取 SAM
local-token cache。已有完整且兼容的 cache 时不会加载 SAM3.1 或 StreamVGGT
backbone，只使用一张卡执行 bounded ICP 的 `cdist`；cache 旧、缺字段或不完整时会自动重建。
V5 angular-Huber ray-centre solver 和其余统计在 CPU 上运行。可以覆盖：

```bash
V75_GPU=1 zsh streaming_couping/commands_v75_explicit_pose_causality.txt
V75_ICP_DEVICE=cpu zsh streaming_couping/commands_v75_explicit_pose_causality.txt
V75_REBUILD_CACHE=1 zsh streaming_couping/commands_v75_explicit_pose_causality.txt
```

cache 缺失或显式设置 `V75_REBUILD_CACHE=1` 时，命令使用三张卡重建 V7.4 动态实例 cache：

```text
StreamVGGT = physical GPU 1,2
SAM3.1     = physical GPU 3
```

可通过 `V75_STREAM_GPU0`、`V75_STREAM_GPU1`、`V75_SAM_GPU` 修改。

## 数据流

```text
V7.4 forward-only SAM3.1 dynamic masks/IDs
  → mask 内冻结 StreamVGGT world points
  → 当前实例点集对历史 object map 的 bounded translation ICP
  → 多实例共同 translation；单实例时选择最可靠 proposal
  → 将 proposal 施加到当前实例 world points
  → V5 angular-Huber point-to-ray camera-centre solver
  → 条件数、ray RMSE、最大 center shift 门控
  → 接受 refined center 或逐帧精确回退 raw StreamVGGT
```

rotation 始终逐元素保留 raw StreamVGGT。本实验只验证旧 V5 已经证明较稳定的解析 center
部分；只有该实验通过，才有理由增加 full-SE(3) solver。

## 时间协议

| fold | 历史前缀 | 未见测试帧 |
|---|---|---|
| short | 90–255 | 270–345 |
| medium | 90–345 | 360–405 |
| long | 90–405 | 420–525 |

每个 history 变体同时运行：

- `frozen`：到 cutoff 后冻结 object map；
- `online`：预测当前测试帧后才允许因果写回，永远不提前读取未来帧。

## 对照

| 变体 | 含义 |
|---|---|
| `raw_streamvggt` | 不运行任何 solver |
| `sam_current_ray` | 当前 SAM mask + ray solver，无历史 ICP |
| `full_image_history` | 全图区域，点数与当前 SAM support 匹配 |
| `sam_bbox_history` | SAM mask 外接框 |
| `random_same_area_history` | 对每个 mask 做保持面积的确定性空间平移 |
| `sam_stale_time_history` | 用上一输入帧的 mask 处理当前 pointmap |
| `sam_history_correct` | 正确 SAM slot 与历史 object map |
| `sam_history_wrong_id` | 将当前 slot 与另一个已出生 slot 的 map 循环错配 |
| `gt_mask_oracle_history` | GT instance mask 上限，只报告，不参与选择 |

除 GT oracle 外，所有区域控制都使用当前 SAM union 的相同有效点数上限。GT oracle 使用自身
有效点数，以区分“solver 本身无能力”和“SAM mask 质量不足”；它不属于公平主结果。

## 输出

```text
outputs/streaming_couping_v75_explicit_pose_causality/
├── v75_explicit_pose_causality.csv
├── v75_explicit_pose_frames.csv
├── v75_dynamic_instance_diagnostics.csv
├── v75_decision_summary.md
├── v75_run_metadata.json
└── cache_audit/
```

主 CSV 每个 fold/variant/memory mode 一行。逐帧表记录 ICP fitness/RMSE、参与实例、ray
残差、proposal/applied center shift、raw/refined error 和 fallback 原因。

所有 `gain_*` 和 pass/fail 字段都使用 mean camera-center error；`pose_loss` 只作为辅助指标，
避免不变的 rotation loss 稀释中心修正。`sam_region_pass=1` 要求正确 SAM history 至少比最好的
full/bbox/random/stale 控制改善 1%。`sam_history_pass=1` 要求它同时至少比 raw、current-only 和
wrong-ID 改善 1%。输出分别汇总 frozen-prefix 和 online-memory 是否在三个 fold 全部通过；
严格总判断要求两种 memory mode 都在三个 fold 全部通过。这样 frozen 结果单独回答“固定历史
是否帮助未来”，online 结果回答“部署时因果更新 memory 后是否持续有效”。

## 结论边界

该实验可以证明或否定：

- SAM mask 是否提供有价值的静态区域选择；
- 正确 SAM persistent identity 是否为未来帧提供额外相机中心约束；
- 提升是否只来自点数、外接框、随机区域或过期 mask。

它不能证明：

- SAM local descriptor/token 改善位姿；
- 相机旋转得到改善；
- dense pointmap 被直接优化；
- 跨场景泛化。
