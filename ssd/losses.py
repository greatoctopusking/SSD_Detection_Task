"""SSD 损失函数（PyTorch 版）。

提供两种实现，由 `build_loss(num_classes, loss_cfg)` 按 yaml 的 `loss.type` 选择：

1) MultiBoxLoss —— **论文原版（默认，推荐）**
   - 定位损失：仅正样本的 SmoothL1(pred_loc, gt_loc)，4 个偏移求和后 / 正样本数；
   - 分类损失：对全部锚点做 **Softmax 交叉熵**（81 类，含背景），再用
     **Hard Negative Mining**：按分类损失降序取负样本 top-k（k = neg_pos_ratio × 正样本数），
     与正样本损失相加后 / 正样本数。
   - 配合推理端的 `softmax` 使用（背景类为 0，丢弃背景后逐类 NMS）。

2) FocalLoss —— 手册版（sigmoid + focal 调制）
   - 与手册 class_loss 语义一致，但把"对锚点求和"后再"对类别取平均"，
     避免 8732×81 项累加 ÷ 少量正样本导致的梯度爆炸（手册原版 loss 量级可达数千）。
   - 配合推理端的 `sigmoid` 使用。

两者 forward 签名一致：criterion(pred_loc, pred_cls, gt_loc, gt_label, num_match)
    pred_loc (B,8732,4) 编码偏移 / pred_cls (B,8732,81) 原始 logits
    gt_loc   (B,8732,4) / gt_label (B,8732) int64（0=背景）/ num_match (B,1) 正样本数
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    """标准 SSD MultiBox 损失：SmoothL1（正样本） + Softmax CE（正样本 + HNM 负样本）。"""

    def __init__(self, num_classes: int = 81, neg_pos_ratio: int = 3, alpha: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.alpha = float(alpha)

    def forward(
        self,
        pred_loc: torch.Tensor,    # (B, 8732, 4)
        pred_cls: torch.Tensor,    # (B, 8732, num_classes) logits
        gt_loc: torch.Tensor,      # (B, 8732, 4)
        gt_label: torch.Tensor,    # (B, 8732) int64, 0=背景
        num_match: torch.Tensor | None = None,   # (B,1) 可选（内部自算正样本数）
    ) -> torch.Tensor:
        pred_loc = pred_loc.float()
        pred_cls = pred_cls.float()
        gt_loc = gt_loc.float()
        gt_label = gt_label.long()
        B, A, _ = pred_loc.shape

        pos_mask = gt_label > 0                                   # (B,A)
        num_pos = pos_mask.sum(dim=1).clamp(min=1).float()        # (B,)

        # ---------- 定位损失（仅正样本，4 个偏移求和后按正样本数归一） ----------
        loc_all = F.smooth_l1_loss(pred_loc, gt_loc, reduction="none", beta=1.0).sum(-1)  # (B,A)
        loc_loss = (loc_all * pos_mask.float()).sum(dim=1) / num_pos                      # (B,)

        # ---------- 分类损失（全部锚点 Softmax CE，稳定实现） ----------
        cls_all = F.cross_entropy(
            pred_cls.transpose(1, 2), gt_label, reduction="none"
        )                                                          # (B,A)
        cls_pos = (cls_all * pos_mask.float()).sum(dim=1)          # (B,)

        # ---------- Hard Negative Mining ----------
        num_neg = torch.min(
            (self.neg_pos_ratio * num_pos).long(),
            (A - pos_mask.sum(dim=1)).clamp(min=0).long(),
        )                                                          # (B,) 每样本负样本配额
        cls_neg = torch.zeros_like(num_pos)
        max_neg = int(num_neg.max().item()) if num_neg.numel() else 0
        if max_neg > 0:
            neg_scores = cls_all.masked_fill(pos_mask, -1e9)       # 正样本排除在外
            topk_vals, _ = torch.topk(neg_scores, k=max_neg, dim=1)  # (B,max_neg)
            rank = torch.arange(max_neg, device=cls_all.device).unsqueeze(0)  # (1,max_neg)
            keep = (rank < num_neg.unsqueeze(1)).float()                     # (B,max_neg)
            cls_neg = (torch.nan_to_num(topk_vals, neginf=0.0) * keep).sum(dim=1)

        cls_loss = (cls_pos + cls_neg) / num_pos                   # (B,)
        total = self.alpha * loc_loss + cls_loss
        return total.mean()


class FocalLoss(nn.Module):
    """手册版损失（sigmoid + focal），仅归一化方式修正（类别维取均值），其余与手册一致。"""

    def __init__(self, num_classes: int = 81, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        pred_loc: torch.Tensor,
        pred_cls: torch.Tensor,
        gt_loc: torch.Tensor,
        gt_label: torch.Tensor,
        num_match: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_loc = pred_loc.float()
        pred_cls = pred_cls.float()
        gt_loc = gt_loc.float()
        gt_label = gt_label.long()

        pos_mask = (gt_label > 0)
        num_pos = pos_mask.sum(dim=1).clamp(min=1).float()

        loc_all = F.smooth_l1_loss(pred_loc, gt_loc, reduction="none", beta=1.0).sum(-1)
        loc_loss = (loc_all * pos_mask.float()).sum(dim=1) / num_pos

        targets = F.one_hot(gt_label, num_classes=self.num_classes).float()
        bce = F.binary_cross_entropy_with_logits(pred_cls, targets, reduction="none")
        sigmoid = torch.sigmoid(pred_cls)
        p_t = targets * sigmoid + (1 - targets) * (1 - sigmoid)
        modulating = torch.pow(1.0 - p_t, self.gamma)
        alpha_w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = modulating * alpha_w * bce                        # (B,A,C)
        cls_loss = (focal.sum(dim=1).mean(dim=1)) / num_pos        # 对锚点求和、类别取平均

        return (loc_loss + cls_loss).mean()


def build_loss(num_classes: int, loss_cfg: dict | None = None) -> nn.Module:
    """按配置构造损失：loss.type = multibox（默认）| focal。"""
    cfg = dict(loss_cfg or {})
    loss_type = str(cfg.get("type", "multibox")).lower()
    if loss_type == "focal":
        return FocalLoss(
            num_classes=num_classes,
            alpha=float(cfg.get("focal_alpha", cfg.get("alpha", 0.75))),
            gamma=float(cfg.get("focal_gamma", cfg.get("gamma", 2.0))),
        )
    if loss_type != "multibox":
        print(f"[loss] 未知 loss.type={loss_type}，回退为 multibox")
    return MultiBoxLoss(
        num_classes=num_classes,
        neg_pos_ratio=int(cfg.get("neg_pos_ratio", 3)),
        alpha=float(cfg.get("alpha", 1.0)),
    )
