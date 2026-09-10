"""验证集 mAP 评估（train.py 训练中途评估 与 eval.py 共用）。

推理端概率：**softmax**（配合 MultiBoxLoss 的 softmax 训练）；若用 focal 训练，
可通过 use_sigmoid=True 切回 sigmoid。COCOMetrics 内部按类别 1..80（丢弃背景类 0）做阈值过滤 + NMS。
"""
from __future__ import annotations

import os

import torch

from ..utils.coco_eval import COCOMetrics
from ..utils.postprocess import decode_boxes


@torch.no_grad()
def evaluate_map(
    net,
    val_loader,
    default_boxes,
    anno_json: str,
    device,
    min_score: float = 0.1,
    nms_threshold: float = 0.6,
    max_boxes: int = 100,
    save_path: str | None = None,
    use_sigmoid: bool = False,
    log_fn=print,
) -> float:
    """在 val_loader 上跑一遍，返回 mAP（IoU=0.50:0.95 的 AP）。"""
    was_training = net.training
    net.eval()
    metrics = COCOMetrics(
        anno_json=anno_json,
        min_score=min_score,
        nms_threshold=nms_threshold,
        max_boxes=max_boxes,
    )

    for img_ids, img, shapes in val_loader:
        img = img.to(device, non_blocking=True)
        pred_loc, pred_cls = net(img)
        if use_sigmoid:
            probs = torch.sigmoid(pred_cls.float())
        else:
            probs = torch.softmax(pred_cls.float(), dim=-1)
        loc_np, sc_np = pred_loc.float().cpu().numpy(), probs.cpu().numpy()
        for i in range(img.shape[0]):
            boxes_pct = decode_boxes(loc_np[i], default_boxes)
            metrics.update(int(img_ids[i]), boxes_pct, sc_np[i], shapes[i].numpy())

    if save_path is None:
        save_path = os.path.join("outputs", "predictions.json")
    mAP = metrics.get_metrics(save_path=save_path)

    if was_training:
        net.train()
    return float(mAP)
