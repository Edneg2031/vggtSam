# T0.2：同场景未见后续时间段确认

## 验证范围

存储不足时，不新增完整场景。T0.2 在发现序列之后选择同一场景的未见时间段：

```text
发现：00a231a370，90–525，step 15
确认：00a231a370，533–591，step 2，共 30 帧
```

确认帧与发现帧不重叠，并且全部晚于 525。这能检验固定规则是否具有
`same-scene temporal generalization`，不能宣称跨场景泛化。

## 完全冻结的规则

T0.1 后确定的 gate 不允许在确认时间段上调整：

```text
positive_depth
AND condition_number < 100
AND num_views >= 4
```

候选生成仍使用与 T0 完全相同的 descriptor、matching、QK history 与
triangulation 参数。三个评分分支为：

```text
correct_persistent_id
foreground_union
shuffled_persistent_id
```

gate assignment 会在读取 holdout 的 GT-derived anchor score 之前冻结并签名。
正式 V0 始终不变：

```text
Pose     = first anchor + QK Top-4
Pointmap = full-history raw pointmap
Semantic = SAM persistent mask / ID
```

## 运行

完整低存储流水线只需一条命令：

```bash
zsh streaming_couping/commands_t02_temporal_holdout.txt
```

命令先用 manifest 检查 30 帧所需文件，再生成隔离的临时 cache 和 QK pose，
运行 T0 锚点探针，随后删除临时 dense cache/QK。最终只保留：

```text
outputs/streaming_couping_t02_sequence_planner/
outputs/streaming_couping_t02_temporal_holdout_anchor_probe/
outputs/streaming_couping_t02_multiview_confirmation/
```

它不会覆盖或删除 `outputs/streaming_couping_v0/`。已有合格 T0 holdout anchors
时，默认直接复用；设置 `T02_REBUILD_ANCHORS=1` 才重新生成。

## 判断

correct-ID 分支同时满足以下条件才是 GO：

```text
anchor_count >= 30
Tri RMSE < Raw RMSE
Tri P90  < Raw P90
```

GO 只表示固定 gate 在同场景后续时间段得到确认，之后才允许设计保守的局部
pointmap refinement。NO-GO 时停止 training-free triangulation refinement，保留
正式 V0。
