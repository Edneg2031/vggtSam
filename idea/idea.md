# 1 任务
输入： RGB序列或者视频、语义text
输出：语义地图点云、mask对应关系
方法结合流式重建的几何特征以及sam3的多目标开放词汇跟踪能力，做外观特征融合


# 2 核心创新点
实例mask + 跨视角mask一致性约束 -> 减少几何层漂移
rgb + 几何层特征 -> 增强跨视角mask一致性判别


# 3 当前想法
## model
rgb 和 prompt 输入 sam3 和 streamVGGT 得到 geometry tokens 和 semantic tokens，
融合两个特征得到 fused tokens，通过一个 head 得到 pointmap[H, W, 3] semantic_feat [H, W, semantic_dim]。
然后 semantic_feat 经过一个分类头 cls_head 进行分类，得到 semantic_mask [H,W,cls_num]这个流程

## 4 trainning
### 4.1 数据准备
```
frames: [B, T, C, H, W] (连续时序图像)
text_prompts: 文本提示词
gt_pointmaps: [B, T, N, 3] (3D 真值)
gt_semantic_masks: [B, T, N] (每个像素/Token对应的类别真值)
gt_instance_ids: [B, T, N] (跨视角同实体对齐真值)
```
### 4.2 loss
gt semantic mask 对应的区域的 pred pointmap 与 gt pointmap 进行损失
gt semantic 和 pred semantic logits 做逐像素的损失


## 5 当前表述
给定无位姿 RGB 视频流与一个文本提示，系统在线输出带持久身份的 2D masklets、统一世界坐标中的 3D pointmap/point cloud、以及 prompt-conditioned 3D instance mask。
