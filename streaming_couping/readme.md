# Streaming Coupling Paired Memory Test

该实验只验证一个变量：**几何修正结果是否写入 SAM3 memory**。
SAM3、StreamVGGT 均冻结，不创建单帧 SAM3 session，不重新检测实例。
两条分支使用独立但相同初始化的原视频 session，并在恢复帧前后采用相同的
分段传播时序；代码会检查恢复前输出一致，避免共享状态或执行顺序污染对照。

## 共同数据流

```text
RGB 序列 + reference GT instance mask
        |                         |
        v                         v
SAM3 原视频 session        StreamVGGT causal geometry
固定 obj_id                 reference instance world points
        |                         |
        |                  投影并与当前 pointmap 检查
        |                         v
        +-----------> 几何 box + 3 个支持正点
                              |
                              v
             在原 session 内修正同一个 SAM3 obj_id
                              |
                              v
                    共同 recovery mask
```

只有 reference GT mask 用于初始化。其他帧 GT 只计算指标，不参与候选生成、
恢复帧选择或 SAM3 修正。

## 唯一消融变量

恢复帧之前，两条分支完全相同；恢复帧使用同一个几何 box、同一组正点和
逐像素相同的校正 mask。box 与点直接修正已有 `obj_id`，不经过 detector；
box 是 SAM3 的提示，不会硬裁剪输出 mask。

- `no_memory`：恢复帧显示修正 mask，但不改变未来 tracker state；未来帧使用
  未写回修正的原 SAM3 轨迹。
- `memory`：将同一个修正结果通过 SAM3 原生 memory encoder 写入同一
  `obj_id`，未来帧从修正后的 memory 继续传播。

代码会检查 recovery frame 的两份 mask 是否逐像素相同。判断 memory 效果只看：

```text
no_memory_post_recovery_iou
memory_post_recovery_iou
```

## 运行

完整服务器命令见 `streaming_couping/commands.txt`。入口为：

```bash
PYTHONPATH=src:. python -m streaming_couping.scripts.run_bridge \
  --config streaming_couping/configs/default.yaml
```

旧 manifest 的 RGB 若仍指向不可读 NAS，先运行 `commands.txt` 中的
`scripts/cache_scannetpp_rgb.py`。`--allow-summary-fallback` 仅适合调试，
不能用于最终定量结果。

## 输出

- `summary.csv`：no-memory、memory 的总体及恢复后指标。
- `frame_metrics.csv`：逐帧候选质量、分支 IoU、score 与恢复帧标记。
- `paired_memory_report.png`：同图逐帧比较 no-memory 与 memory。
- `resolved_config.json`：实际运行配置。

`summary.csv` 还会显式记录 `same_obj_id=1`、`redetection_used=0`、
`paired_causal_split=1`，便于检查对照条件。

## 当前边界

- 当前只测试一次 aligned geometry correction 对后续 memory 的影响。
- 实例身份始终由同一个 SAM3 session 和同一个 `obj_id` 维持。
- 不包含重新检测、点云地图更新、相机优化或联合训练。
