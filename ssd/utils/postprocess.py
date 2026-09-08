"""推理后处理 —— 对应手册 SsdInferWithDecoder（解码）+ 逐类 NMS 过滤（COCOMetrics.update 前的部分）。

解码公式（与手册一致）：
    pred_xy = pred_loc[:2] * 0.1 * anchor_wh + anchor_xy
    pred_wh = exp(pred_loc[2:] * 0.2) * anchor_wh
    → tlbr，clip 到 [0, 1]（百分比坐标域，供评估时按原图尺寸缩放）
"""
from __future__ import annotations

import numpy as np

from .nms import nms


def decode_boxes(
    pred_loc: np.ndarray,
    default_boxes: np.ndarray,
    var_xy: float = 0.1,
    var_wh: float = 0.2,
) -> np.ndarray:
    """pred_loc: (8732,4) 或 (B,8732,4)；default_boxes: (8732,4) [cy,cx,h,w] 百分比。
    返回 tlbr [y1,x1,y2,x2] 百分比，已 clip 到 [0,1]。"""
    loc = np.asarray(pred_loc, dtype=np.float32)
    single = loc.ndim == 2
    if single:
        loc = loc[np.newaxis]
    db = np.asarray(default_boxes, dtype=np.float32)[np.newaxis, :, :]  # (1,8732,4)

    pred_xy = loc[:, :, :2] * var_xy * db[:, :, 2:] + db[:, :, :2]      # (B,8732,2) cy,cx
    pred_wh = np.exp(loc[:, :, 2:] * var_wh) * db[:, :, 2:]             # (B,8732,2) h,w
    boxes = np.concatenate([pred_xy - pred_wh / 2.0, pred_xy + pred_wh / 2.0], axis=2)
    boxes = np.clip(boxes, 0.0, 1.0)                                    # [y1,x1,y2,x2]
    return boxes[0] if single else boxes


def filter_predictions(
    boxes_percent: np.ndarray,      # (8732,4) tlbr 百分比
    scores: np.ndarray,             # (8732, num_classes) sigmoid 后得分
    image_shape,                    # (h, w) 原图尺寸
    min_score: float = 0.1,
    nms_threshold: float = 0.6,
    max_boxes: int = 100,
):
    """逐类（1..num_classes-1）阈值过滤 + NMS。
    返回按分数降序的检测列表：每项 dict {category, score, box_px, box_percent}，
    box_px 为像素域 [y1,x1,y2,x2]。"""
    num_classes = scores.shape[1]
    h, w = float(image_shape[0]), float(image_shape[1])
    dets = []
    for c in range(1, num_classes):
        sc = scores[:, c]
        keep = sc > min_score
        if not keep.any():
            continue
        boxes_px = boxes_percent[keep] * np.array([h, w, h, w], dtype=np.float32)
        kept_idx = nms(boxes_px, sc[keep], thres=nms_threshold, max_boxes=max_boxes)
        for i in kept_idx:
            dets.append(
                {
                    "category": int(c),
                    "score": float(sc[keep][i]),
                    "box_px": [float(v) for v in boxes_px[i]],      # y1,x1,y2,x2 像素
                    "box_percent": [float(v) for v in boxes_percent[keep][i]],
                }
            )
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets
