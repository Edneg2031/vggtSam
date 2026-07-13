# SAM3 Token 到 StreamVGGT 的第一阶段验证

本目录只做诊断，不训练融合网络，也不修改 StreamVGGT KV cache。

固定五帧序列上同时导出：

- SAM3 tracker `fpn2`：`[T, 256, 72, 72]`；
- SAM3 实际传播与 GT-visible oracle 分支的 spatial memory；
- SAM3 mask decoder object pointer：`[T, 256]`；
- StreamVGGT aggregator 第 `4/11/17/23` 层 patch tokens。

查询帧中 instance mask 内的 token 均值作为 object prototype。脚本计算各特征在其余帧中的余弦相似度、Top-K 实例命中率、正负区域 margin 和 anchor 定位误差。

主要输出：

- `feature_summary.csv`：判断哪一层最适合作为跨视角 identity signal；
- `feature_frame_metrics.csv`：逐帧指标；
- `object_pointer_metrics.csv`：SAM3 object pointer 的跨帧稳定性；
- `similarity_maps/*.png`：各特征的跨视角响应热图；
- `sam3_tracker_masks.png`：实际 SAM3 memory 流程的掩码结果。

只有当某类 SAM3 token 在跨视角可见帧中稳定命中目标，且优于原始 StreamVGGT token，下一步才把它作为 query/identity token 注入 StreamVGGT；否则先不改 StreamVGGT memory。
