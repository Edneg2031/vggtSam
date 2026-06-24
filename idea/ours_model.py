import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentGeometricSemanticModel(nn.Module):
    def __init__(self, d_fuse=512, num_classes=80):
        super().__init__()
        # 1. 基础大模型骨干 (实际中可挂载 LoRA)
        self.stream_vggt_backbone = self._load_vggt_backbone() # 冻结或开启 LoRA
        self.sam3_backbone = self._load_sam3_backbone()         # 冻结或开启 LoRA
        
        # 2. 维度投影层 (将不同大模型的 Token 映射到统一隐空间)
        self.proj_vggt = nn.Linear(vggt_dim, d_fuse)
        self.proj_cam = nn.Linear(cam_dim, d_fuse)
        self.proj_sam = nn.Linear(sam3_dim, d_fuse)
        
        # 3. 隐空间交叉注意力融合层 (融合几何、相机与语义 Token)
        # 允许语义 Token 作为 Query 去检索几何与相机位姿先验
        self.cross_attention_fusion = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_fuse)
        
        # 4. 多任务解码器 Heads
        # Head A: 稠密 3D 坐标 + 语义分类直出
        self.pointmap_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.ReLU(),
            nn.Linear(d_fuse, 3 + d_fuse) # 输出 3D 坐标 (X,Y,Z) + 丰富的语义特征向量
        )
        self.classifier_head = nn.Linear(d_fuse, num_classes) # 开放词汇分支亦可替换为 Text Embedding 对齐
        
        # Head B: 跨视角掩膜匹配 (利用特征相关性计算 Cross-View Mask Match)
        self.match_query_proj = nn.Linear(d_fuse, d_fuse)
        self.match_key_proj = nn.Linear(d_fuse, d_fuse)

    def forward(self, image, text_prompts, kv_cache=None):
        """
        单帧流式前向传播
        """
        # Step A: 提取大模型特征
        # vggt_feat: [B, N_vggt, D_vggt], cam_token: [B, N_cam, D_cam]
        vggt_feat, cam_token, updated_kv_cache = self.stream_vggt_backbone(image, kv_cache)
        # sam_feat: [B, N_sam, D_sam] (紧跟文本提示提取的视频跟踪 Token)
        sam_feat = self.sam3_backbone(image, text_prompts)
        
        # Step B: 投影至统一通道维度
        f_vggt = self.proj_vggt(vggt_feat)
        f_cam = self.proj_cam(cam_token)
        f_sam = self.proj_sam(sam_feat)
        
        # Step C: 隐空间交叉注意力机制融合
        # 让具有长程跟踪记忆的 SAM 特征充当 Query，强行吸收当前帧的几何和相机位姿
        geometry_context = torch.cat([f_vggt, f_cam], dim=1) # [B, N_vggt + N_cam, d_fuse]
        f_fused, _ = self.cross_attention_fusion(query=f_sam, key=geometry_context, value=geometry_context)
        f_fused = self.fusion_norm(f_fused + f_sam) # 残差连接与归一化
        
        # Step D: 多任务解码输出
        # 1. 3D 语义点云分支
        geo_outputs = self.pointmap_head(f_fused)
        pred_pointmap = geo_outputs[..., :3]           # [B, N_sam, 3] 绝对三维坐标
        semantic_embeddings = geo_outputs[..., 3:]     # [B, N_sam, d_fuse]
        pred_logits = self.classifier_head(semantic_embeddings) # [B, N_sam, Num_Classes]
        
        return {
            "pred_pointmap": pred_pointmap,
            "pred_logits": pred_logits,
            "fused_tokens": f_fused, # 吐出融合 Token，用于后续时序跨视角匹配
            "kv_cache": updated_kv_cache
        }

    def compute_mask_correspondence(self, fused_tokens_curr, fused_tokens_hist):
        """
        输入当前帧与历史帧的融合 Token，直出跨视角的 Mask 匹配概率矩阵
        """
        q = self.match_query_proj(fused_tokens_curr) # [B, N_sam, d_fuse]
        k = self.match_key_proj(fused_tokens_hist)   # [B, N_sam, d_fuse]
        # 计算相关性内积矩阵
        match_matrix = torch.matmul(q, k.transpose(-1, -2)) # [B, N_sam, N_sam]
        return torch.sigmoid(match_matrix)