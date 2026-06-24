import torch
from torch.utils.data import DataLoader

def train_epoch(model, dataset, optimizer, lr_scheduler, device):
    model.train()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=seq_collate)
    
    for batch_idx, video_sequence in enumerate(dataloader):
        optimizer.zero_grad()
        
        # video_sequence 包含: 
        # frames: [B, T, C, H, W] (连续时序图像)
        # text_prompts: 文本提示词
        # gt_pointmaps: [B, T, N, 3] (3D 真值)
        # gt_semantic_masks: [B, T, N] (每个像素/Token对应的类别真值)
        # gt_instance_ids: [B, T, N] (跨视角同实体对齐真值)
        
        frames = video_sequence["frames"].to(device)
        gt_pointmaps = video_sequence["gt_pointmaps"].to(device)
        gt_semantics = video_sequence["gt_semantic_masks"].to(device)
        gt_instances = video_sequence["gt_instance_ids"].to(device)
        text_prompts = video_sequence["text_prompts"]
        
        B, T, _, _, _ = frames.shape
        
        # 初始化流式推理的 KV 缓存与历史 Token 记忆库
        kv_cache = None
        historical_token_buffer = [] 
        
        sequence_loss = 0.0
        
        # 开始沿时间步（流式输入）向前推进
        for t in range(T):
            current_frame = frames[:, t, ...] # 取出时刻 t 的单帧图像
            
            # 1. 前向传播：混合几何与追踪语义
            outputs = model(current_frame, text_prompts, kv_cache=kv_cache)
            
            pred_pointmap = outputs["pred_pointmap"]
            pred_logits = outputs["pred_logits"]
            fused_tokens = outputs["fused_tokens"]
            kv_cache = outputs["kv_cache"] # 更新缓存供下一步 t+1 使用
            
            # 2. 计算当前帧的瞬时几何与语义损失
            loss_3d = F.l1_loss(pred_pointmap, gt_pointmaps[:, t, ...])
            loss_cls = F.cross_entropy(pred_logits.transpose(1, 2), gt_semantics[:, t, ...])
            
            current_loss = 1.0 * loss_3d + 0.5 * loss_cls
            
            # 3. 计算跨视角长程 Mask 匹配损失 (反哺核心)
            if t > 0:
                loss_match = 0.0
                # 随机抽取一个历史帧，强迫网络跨越时间进行特征对齐匹配
                hist_t = np.random.randint(0, t)
                fused_tokens_hist = historical_token_buffer[hist_t]
                
                # 预测当前帧 Token 与选中历史帧 Token 的匹配关系矩阵
                pred_match_matrix = model.compute_mask_correspondence(fused_tokens, fused_tokens_hist)
                
                # 构造真值匹配矩阵: 如果当前时刻 t 的 Token i 与历史时刻 hist_t 的 Token j 的 Instance_ID 相同，则为 1
                gt_match_matrix = (gt_instances[:, t, :, None] == gt_instances[:, hist_t, None, :]).float()
                
                loss_match = F.binary_cross_entropy(pred_match_matrix, gt_match_matrix)
                current_loss += 0.8 * loss_match
                
            # 将当前帧融合后的特征送入记忆库，供后续帧做跨视角匹配训练
            historical_token_buffer.append(fused_tokens.detach())
            
            # 累加整个时序序列的损失
            sequence_loss += current_loss
            
        # 4. 整个序列流式推进完成后，进行统一反向传播与参数更新
        # 此时梯度会沿着交叉注意力、投影层回传，迫使网络学习“如何用几何稳定语义，用语义锚定几何”
        sequence_loss = sequence_loss / T
        sequence_loss.backward()
        
        # 梯度裁剪，防止大模型微调梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Sequence Mean Loss: {sequence_loss.item():.4f}")
            
    lr_scheduler.step()