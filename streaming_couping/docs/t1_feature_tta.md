# T1：Full-History Teacher → Masked-History Student TTA

## 目的

只验证一件事：不使用 GT 训练时，full-history StreamVGGT 的中间特征能否通过小型 LoRA，让 held-out 12 帧的 dense pointmap 超过正式 V0 raw full-history pointmap。

正式 V0 始终不变：

```text
Pose     = QK Top-4
Pointmap = raw full-history StreamVGGT
Semantic = SAM persistent mask / ID
```

## 固定协议

- adaptation：前 18 帧，frame `90..345`；
- evaluation：后 12 帧，frame `360..525`；
- teacher：冻结的 full-causal-history `[4, 11, 17, 23]` 层 global patch token；
- student：anchor/current 保留，中间 history 以 `p=0.5, seed=0` 确定性 dropout；
- LoRA：仅 global-attention 的 `qkv/proj`，rank 4、alpha 4；
- loss：四层 current-frame global patch token cosine consistency；
- optimizer：AdamW，`lr=1e-4`、weight decay 0、5 epochs，共 85 steps/branch；
- 最终 pointmap：LoRA 冻结后重新使用完整 causal history；
- 历史 KV 在每次 prefix 内作为 stop-gradient context，当前帧路径回传梯度；每次更新后从头重放 prefix，避免陈旧缓存和整段 BPTT 显存膨胀。

对照固定为：

```text
A. raw_full_history
B. correct_teacher_lora
C. shuffled_teacher_lora
D. zero_update_lora
```

`prepare → adapt → score` 是三个独立进程。`adapt` 只能读取不含 raw pointmap/GT/SAM 的 clean artifact；`score` 先冻结并哈希 candidate artifact，之后才读取 V0 raw pointmap 和 GT。

其中 `raw_full_history` 不会复制进 candidate artifact，而是在 score 阶段直接从只读 V0 cache 加入，避免额外存一份正式 baseline。

## 一键运行

```bash
zsh streaming_couping/commands_t1_feature_tta.txt
```

如果 V0 cache 已删除，这条命令会自动只重建 T1 所需的 V0 cache；如果 QK pose artifact 也不存在，则只额外重建 QK 分支，不会运行 V0 audit 和语义地图导出。实验成功后，命令会自动删除本次临时重建的大 cache。

默认使用物理 GPU 1、2。可覆盖：

```bash
T1_FEATURE_TTA_GPU0=0 T1_FEATURE_TTA_GPU1=1 \
zsh streaming_couping/commands_t1_feature_tta.txt
```

clean artifact 体积较大，评分完成后默认删除；如需保留：

```bash
T1_KEEP_CLEAN_ARTIFACT=1 \
zsh streaming_couping/commands_t1_feature_tta.txt
```

如果希望同时保留自动重建的 V0 cache：

```bash
T1_KEEP_REBUILT_V0_CACHE=1 \
zsh streaming_couping/commands_t1_feature_tta.txt
```

## GO / NO-GO

GO 必须同时满足：

- zero-update pointmap/confidence 等价检查通过；
- 正式 QK pose artifact 的 SHA256 前后一致；
- correct LoRA RMSE 小于 raw；
- correct LoRA P90 不差于 raw；
- 12 帧中至少 7 帧 RMSE 改善；
- correct LoRA RMSE 小于 shuffled teacher LoRA。

否则为 NO-GO，停止继续堆 feature consistency loss，正式 V0 保持不变。

主要输出：

```text
outputs/streaming_couping_t1_feature_tta/summary.json
outputs/streaming_couping_t1_feature_tta/copyable_result.txt
outputs/streaming_couping_t1_feature_tta/branch_summary.csv
outputs/streaming_couping_t1_feature_tta/frame_metrics.csv
outputs/streaming_couping_t1_feature_tta/history_dropout.csv
outputs/streaming_couping_t1_feature_tta/lora_states.pt
outputs/streaming_couping_t1_feature_tta/candidate_pointmaps.pt
```
