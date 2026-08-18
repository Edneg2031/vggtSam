# T0.2：可靠多视图三角化的跨序列确认

## 固定假设

T0.2 只接受以下 gate，不允许在确认序列上调参：

```text
positive_depth
AND condition_number < 100
AND num_views >= 4
```

它不使用 reprojection error 或 angle threshold。正式 V0 始终保持 QK pose、
full-history raw pointmap 和 SAM persistent semantics。

## 新序列前置条件

必须选择未参与 T0/T0.1 gate 设计的新 clip，并用与 T0 相同的候选协议生成：

```text
outputs/streaming_couping_t02_new_sequence_anchor_probe/
  anchors_candidate.pt
  anchor_scores.csv
  summary.json
```

三个必需分支是：

```text
correct_persistent_id
foreground_union
shuffled_persistent_id
```

T0.2 会拒绝旧 clip `00a231a370_90_525_step15_37_68_54`。它先对三个分支写入
相同的冻结 gate 和协议签名，之后才读取新序列的 GT-derived anchor scores。

## 运行与判断

```bash
zsh streaming_couping/commands_t02_multiview_confirmation.txt
```

只有 correct-ID 分支同时满足以下条件才是 GO：

```text
anchor_count >= 30
Tri RMSE < Raw RMSE
Tri P90 < Raw P90
```

GO 后才允许设计 conservative local pointmap refinement；NO-GO 时停止
training-free triangulation pointmap refinement，并保留正式 V0。
