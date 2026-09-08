"""SSD 损失 —— 对应手册损失部分。

- 定位损失: SmoothL1(pred_loc, gt_loc) * 正样本 mask，按匹配数归一
- 置信度损失: binary_cross_entropy_with_logits + focal 调制((1-p_t)^2, alpha=0.75)
  （手册 class_loss 语义）
- hard negative mining: 负样本按置信度误差降序取 top-k，负:正 ≈ 3:1
TODO(step 05): 实现 SSDLoss(nn.Module)。
"""
