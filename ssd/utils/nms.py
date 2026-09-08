"""非极大值抑制 —— 对应手册 apply_nms（推理阶段使用，训练不用）。"""
from __future__ import annotations

import numpy as np


def nms(
    all_boxes: np.ndarray,     # (N,4) [y1, x1, y2, x2]（像素或百分比均可，需同域）
    all_scores: np.ndarray,    # (N,)
    thres: float = 0.6,
    max_boxes: int = 100,
) -> list:
    """按置信度降序贪心选取；删除与已选框 IoU > thres 的候选；达到 max_boxes 即停。返回保留索引列表。"""
    if all_boxes.shape[0] == 0:
        return []
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = all_scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if len(keep) >= max_boxes:
            break
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[rest] - inter)
        order = rest[np.where(ovr <= thres)[0]]
    return keep
