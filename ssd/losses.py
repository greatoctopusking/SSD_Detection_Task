"""SSD 损失 —— 对应手册「损失函数」章节（PyTorch 版）。

总损失（逐样本） = (L_loc + L_conf) / num_match，再对 batch 取平均。

- L_loc ：仅正样本锚点（label>0）的 SmoothL1(pred_loc, gt_loc)，sum 覆盖 4 个偏移；
- L_conf：与手册 class_loss 语义一致 —— sigmoid + BCE，再乘 focal 调制
          ((1 - p_t)^γ, γ=2) 与类别权重 (α=0.75 正类 / 0.25 负类)，对所有 8732 锚点 × 81 类求和。
  说明：背景锚点的 label=0 → 目标为全 0 → 学会压低所有类别得分；focal 自动抑制易分负样本，
  因此无需显式 hard negative mining（手册代码亦如此，未显式实现）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDLoss(nn.Module):
    def __init__(self, num_classes: int = 81, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_loc: torch.Tensor,    # (B, 8732, 4)
        pred_label: torch.Tensor,  # (B, 8732, num_classes) logits
        gt_loc: torch.Tensor,      # (B, 8732, 4) 编码后
        gt_label: torch.Tensor,    # (B, 8732) int64，0=背景
        num_match: torch.Tensor,   # (B, 1) int —— 每样本正样本锚点数
    ) -> torch.Tensor:
        batch = pred_loc.shape[0]

        # ---------- 定位损失（仅正样本） ----------
        mask = (gt_label > 0).float().unsqueeze(-1)          # (B,8732,1)
        smooth_l1 = F.smooth_l1_loss(pred_loc, gt_loc, reduction="none")
        loss_loc = (smooth_l1 * mask).sum(dim=(1, 2))        # (B,)

        # ---------- 置信度损失（focal + BCE，对应手册 class_loss） ----------
        targets = F.one_hot(gt_label.long(), num_classes=self.num_classes).float()  # (B,8732,81)
        bce = F.binary_cross_entropy_with_logits(pred_label, targets, reduction="none")
        sigmoid = torch.sigmoid(pred_label)
        p_t = targets * sigmoid + (1 - targets) * (1 - sigmoid)   # (B,8732,81)
        modulating = torch.pow(1.0 - p_t, self.gamma)
        alpha_w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss_cls = (modulating * alpha_w * bce).sum(dim=1).mean(dim=1)    # (B,)

        # ---------- 归一化与聚合 ----------
        n = num_match.float().squeeze(-1).clamp(min=1.0)          # (B,) 防除零
        total = (loss_loc + loss_cls) / n                         # 每样本
        return total.mean()
