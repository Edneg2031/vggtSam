# Streaming Coupling

当前目录只保留一个正式实验：验证几何恢复得到的同一张 mask 写入 SAM3
原 `obj_id` 的 memory 后，是否改善后续跨视角跟踪。

SAM3 和 StreamVGGT 均冻结。实验不训练 adapter，不做相机位姿优化、ICP、
点云融合或 token 融合。

## 当前测试设置

- 场景：ScanNet++ `00a231a370`
- 帧：`90 105 119 130 140 210 240`
- 实例：`37 cabinet`、`68 wardrobe`
- reference：序列位置 `0`，即帧 `90`
- 暂不测试 `54 bed`：目标面积过大且首帧只局部可见，容易把实例尺度因素混入
  memory writeback 消融

两个实例共享一次 StreamVGGT 几何提取，但分别使用独立的 SAM3 session、
persistent `obj_id`、恢复事件和指标。

## 四个分支

1. `original`
2. `geometry_recovery_no_memory`
3. `geometry_recovery_same_id_memory`
4. `shuffled_geometry_same_id_memory`

`geometry_recovery_no_memory` 和 `geometry_recovery_same_id_memory` 在同一恢复帧
使用逐像素完全相同的 dense mask。唯一变量是该 mask 是否通过 SAM3 原生
memory encoder 写入原 persistent `obj_id`。

`shuffled_geometry_same_id_memory` 固定 reference 几何，循环打乱其余帧的
StreamVGGT 输出；RGB、SAM3 文本候选、阈值和 reference mask 均不变。该分支
用于排除任意几何扰动也能带来改善的可能。

## 因果数据流

```text
reference GT mask
      |
      +----> 原始 SAM3 tracker ----------------------> original
      |
      +----> reference 实例 pointmap
                    |
                    v
          投影到后续帧并检查当前 pointmap 支持
                    |
                    v
            首个“tracker 弱 + geometry 通过”帧
                    |
                    v
          单帧 global-text SAM3 dense candidates
                    |
                    v
             geometry 选择同一张恢复 mask
                    |
              +-----+-----+
              |           |
       只替换当前显示   写入原 obj_id memory
              |           |
              v           v
          no-memory     same-ID memory
```

后续帧 GT mask 只用于计算指标和输出 oracle 诊断列，不参与恢复帧触发、候选
排序或 mask 写回。单帧 text query 产生的临时 ID 只用于读取候选 mask，不会
替换视频 tracker 的 persistent `obj_id`。

代码运行时会检查：

- 因果切分但不写 memory 时，所有原始 SAM3 mask 必须保持不变；
- aligned 两分支在恢复前必须一致；
- aligned 两分支在恢复帧必须使用完全相同的 mask；
- memory 分支写回前后必须保持原 `obj_id`。

任何检查失败都会直接报错，不继续产出可被误解的结果。

## 运行

服务器完整命令保存在 `commands.txt`：

```bash
PYTHONPATH=src:. python -m streaming_couping.scripts.run_recovery_writeback_ablation \
  --config streaming_couping/configs/default.yaml \
  --manifest data/processed/scannetpp_pinhole_2d/manifest.json \
  --scene-id 00a231a370 \
  --instance-ids 37 68 \
  --frame-indices 90 105 119 130 140 210 240 \
  --reference-sequence-index 0 \
  --sam3-device cuda:3 \
  --geometry-device cuda:1 \
  --output-dir outputs/streaming_couping_recovery_writeback_37_68
```

本地 Mac 没有 PyTorch/GPU，只做静态检查；正式数值结果在服务器环境运行。

## 输出

输出根目录包含：

- `summary.csv`：每个实例、每个分支的总体与 recovery 后指标
- `frame_metrics.csv`：逐实例、逐分支、逐帧 IoU 和漏检情况
- `candidate_diagnostics.csv`：候选排序证据及仅供诊断的 GT IoU
- `metadata.json`：实际分支、几何 permutation、reference 和恢复事件

每个 `instance_<id>/` 目录还包含对应的三份 CSV，以及
`recovery_writeback_report.png` 可视化。

优先查看以下比较：

```text
geometry_recovery_same_id_memory.post_recovery_iou
  versus
geometry_recovery_no_memory.post_recovery_iou
```

同时要求 aligned 优于 shuffled，且 `same_obj_id_as_original=1`。恢复帧本身的
IoU 不能证明 memory 有效，因为 aligned 两分支在该帧被刻意设为完全相同；
核心证据必须来自恢复后的可见帧。

## 当前代码结构

```text
scripts/run_recovery_writeback_ablation.py  CLI 入口
src/recovery_writeback_ablation.py          四分支编排、指标和报告
src/recovery.py                             几何恢复挖掘和 mask 指标
src/backbones/sam3_wrapper.py               跟踪、文本候选、因果切分、same-ID 写回
src/backbones/streamvggt_wrapper.py         冻结几何提取
src/aggregation/                            reference 对象点与重访候选
src/bridge/gating.py                        弱跟踪门控和 IoU
src/config.py                               当前实验配置
src/types.py                                当前实验数据结构
```

`docs/method.md` 和 `docs/thread_handoff.md` 仅保留研究过程与历史决策，不代表
当前可运行入口；代码和实验设置以本 README、`commands.txt` 为准。
